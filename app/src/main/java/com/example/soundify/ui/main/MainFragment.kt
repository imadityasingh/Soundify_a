package com.example.soundify.ui.main

import android.content.Intent
import android.graphics.*
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.speech.tts.TextToSpeech
import android.text.method.ScrollingMovementMethod
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.*
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.FileProvider
import androidx.fragment.app.Fragment
import androidx.navigation.fragment.findNavController
import com.example.soundify.R
import com.example.soundify.databinding.FragmentMainBinding
import com.example.soundify.ml.DetectedObject
import com.google.firebase.auth.FirebaseAuth
import org.json.JSONObject
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLEncoder
import java.text.SimpleDateFormat
import java.util.*

class MainFragment : Fragment() {
    private var _binding: FragmentMainBinding? = null
    private val binding get() = _binding!!

    companion object {
        private const val REQUEST_IMAGE_CAPTURE = 1
        private const val REQUEST_IMAGE_PICK = 2
    }

    lateinit var auth: FirebaseAuth
    lateinit var takePictureButton: Button
    lateinit var launchGalleryButton: Button
    lateinit var playButton: Button
    lateinit var pauseButton: Button
    lateinit var imageView: ImageView
    lateinit var resultText: TextView
    lateinit var photoURI: Uri
    lateinit var bitmap: Bitmap
    private var currentPhotoPath: String = ""

    lateinit var textToSpeech: TextToSpeech

    val paint = Paint()
    var detectedObjects = mutableListOf<DetectedObject>()
    lateinit var labels: List<String>
    lateinit var interpreter: Interpreter

    val imageProcessor = ImageProcessor.Builder()
        .add(ResizeOp(300, 300, ResizeOp.ResizeMethod.BILINEAR))
        .build()

    private val getContent = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        uri?.let {
            try {
                val source = ImageDecoder.createSource(requireActivity().contentResolver, it)
                bitmap = ImageDecoder.decodeBitmap(source)
                imageView.setImageBitmap(bitmap)
                getPredictions()
            } catch (e: Exception) {
                Toast.makeText(requireContext(), "Error loading image: ${e.message}", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private val takePicture = registerForActivityResult(ActivityResultContracts.TakePicture()) { success ->
        if (success) {
            try {
                val file = File(currentPhotoPath)
                val source = ImageDecoder.createSource(requireActivity().contentResolver, Uri.fromFile(file))
                bitmap = ImageDecoder.decodeBitmap(source)
                imageView.setImageBitmap(bitmap)
                getPredictions()
            } catch (e: Exception) {
                Toast.makeText(requireContext(), "Error loading image: ${e.message}", Toast.LENGTH_SHORT).show()
            }
        }
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentMainBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        auth = FirebaseAuth.getInstance()

        try {
            // Initialize ML model
            labels = FileUtil.loadLabels(requireContext(), "labels.txt")
            val modelFile = FileUtil.loadMappedFile(requireContext(), "ssd_mobilenet_v1_1_metadata_1.tflite")
            interpreter = Interpreter(modelFile)
        } catch (e: Exception) {
            Toast.makeText(requireContext(), "Error initializing model: ${e.message}", Toast.LENGTH_LONG).show()
        }

        // Initialize views
        takePictureButton = binding.button
        launchGalleryButton = binding.button2
        imageView = binding.imageView
        resultText = binding.result
        playButton = binding.playButton
        pauseButton = binding.pauseButton

        resultText.movementMethod = ScrollingMovementMethod()

        // Set up click listeners
        takePictureButton.setOnClickListener { dispatchTakePictureIntent() }
        launchGalleryButton.setOnClickListener {
            getContent.launch("image/*")
        }

        // Initialize TextToSpeech
        textToSpeech = TextToSpeech(requireContext()) { status ->
            if (status == TextToSpeech.SUCCESS) {
                textToSpeech.language = Locale.US
            }
        }

        // Play button to start speaking detected object
        playButton.setOnClickListener {
            val text = resultText.text.toString()
            if (text.isNotEmpty()) {
                textToSpeech.speak(text, TextToSpeech.QUEUE_FLUSH, null, null)
            }
        }

        // Pause button to stop speaking
        pauseButton.setOnClickListener {
            if (textToSpeech.isSpeaking) {
                textToSpeech.stop()
            }
        }

        // Set up logout button
        binding.logoutButton.setOnClickListener {
            auth.signOut()
            Toast.makeText(requireContext(), "Logged out successfully", Toast.LENGTH_SHORT).show()
            findNavController().navigate(R.id.action_main_to_login)
        }

        // Set up touch listener for image view
        imageView.setOnTouchListener { _, event ->
            if (event.action == android.view.MotionEvent.ACTION_DOWN) {
                val x = event.x
                val y = event.y

                for (detectedObject in detectedObjects) {
                    if (detectedObject.boundingBox.contains(x, y)) {
                        fetchObjectInformation(detectedObject.label)
                        return@setOnTouchListener true
                    }
                }
            }
            false
        }
    }

    private fun dispatchTakePictureIntent() {
        val photoFile: File? = try {
            createImageFile()
        } catch (ex: IOException) {
            null
        }
        if (photoFile != null) {
            photoURI = FileProvider.getUriForFile(
                requireContext(),
                "com.example.soundify.fileprovider",
                photoFile
            )
            takePicture.launch(photoURI)
        }
    }

    @Throws(IOException::class)
    private fun createImageFile(): File {
        val timeStamp: String =
            SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
        val storageDir: File? = requireActivity().getExternalFilesDir(null)
        return File.createTempFile("JPEG_${timeStamp}_", ".jpg", storageDir).apply {
            currentPhotoPath = absolutePath
        }
    }

    private fun getPredictions() {
        if (!::interpreter.isInitialized) {
            Toast.makeText(requireContext(), "Model not initialized yet", Toast.LENGTH_SHORT).show()
            return
        }

        detectedObjects.clear()
        val image = TensorImage.fromBitmap(bitmap)
        val processedImage = imageProcessor.process(image)
        
        // Prepare input and output buffers
        val inputBuffer = processedImage.buffer
        val outputLocations = Array(1) { Array(1917) { Array(4) { 0f } } }
        val outputClasses = Array(1) { Array(1917) { 0f } }
        val outputScores = Array(1) { Array(1917) { 0f } }
        
        // Run inference
        interpreter.run(inputBuffer, arrayOf(outputLocations, outputClasses, outputScores))

        val mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(mutableBitmap)
        val h = mutableBitmap.height
        val w = mutableBitmap.width

        paint.textSize = h / 15f
        paint.strokeWidth = h / 85f

        // Process detections
        for (i in 0 until 1917) {
            val score = outputScores[0][i]
            if (score > 0.5) {
                val boundingBox = RectF(
                    outputLocations[0][i][1] * w,
                    outputLocations[0][i][0] * h,
                    outputLocations[0][i][3] * w,
                    outputLocations[0][i][2] * h
                )
                val label = labels[outputClasses[0][i].toInt()]
                detectedObjects.add(DetectedObject(label, score, boundingBox))
            }
        }

        // Draw bounding boxes and labels
        detectedObjects.forEach { obj ->
            paint.color = Color.RED
            paint.style = Paint.Style.STROKE
            canvas.drawRect(obj.boundingBox, paint)

            paint.color = Color.RED
            paint.style = Paint.Style.FILL
            canvas.drawText(
                "${obj.label} ${String.format("%.2f", obj.confidence)}",
                obj.boundingBox.left,
                obj.boundingBox.top - 10,
                paint
            )
        }

        imageView.setImageBitmap(mutableBitmap)
        resultText.text = detectedObjects.joinToString("\n") { obj ->
            "${obj.label} (${String.format("%.2f", obj.confidence)})"
        }
    }

    private fun fetchObjectInformation(query: String) {
        Thread {
            try {
                val encodedQuery = URLEncoder.encode(query, "UTF-8")
                val urlStr = "https://en.wikipedia.org/api/rest_v1/page/summary/$encodedQuery"
                val url = URL(urlStr)
                val connection = url.openConnection() as HttpURLConnection
                connection.requestMethod = "GET"
                connection.connectTimeout = 5000
                connection.readTimeout = 5000
                val responseCode = connection.responseCode
                if (responseCode == HttpURLConnection.HTTP_OK) {
                    val stream = connection.inputStream
                    val result = stream.bufferedReader().use { it.readText() }
                    val jsonObj = JSONObject(result)
                    val extract = jsonObj.optString("extract", "No information available.")
                    requireActivity().runOnUiThread {
                        resultText.text = extract
                        textToSpeech.speak(extract, TextToSpeech.QUEUE_FLUSH, null, null)
                    }
                } else {
                    requireActivity().runOnUiThread {
                        resultText.text = "Error: $responseCode"
                    }
                }
            } catch (e: Exception) {
                e.printStackTrace()
                requireActivity().runOnUiThread {
                    resultText.text = "Error fetching information."
                }
            }
        }.start()
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
        if (::interpreter.isInitialized) {
            interpreter.close()
        }
        textToSpeech.stop()
        textToSpeech.shutdown()
    }
} 