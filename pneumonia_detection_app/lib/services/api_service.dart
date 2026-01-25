import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:file_picker/file_picker.dart';

class ApiService {
  // Change this to your backend URL
  // For local development: http://localhost:5000
  // For production: your deployed backend URL
  static const String baseUrl = 'http://localhost:5000';

  /// Predict pneumonia from chest X-ray image
  static Future<PredictionResponse> predictPneumonia(PlatformFile file) async {
    try {
      final uri = Uri.parse('$baseUrl/predict');

      var request = http.MultipartRequest('POST', uri);

      // Add the image file
      request.files.add(
        http.MultipartFile.fromBytes('image', file.bytes!, filename: file.name),
      );

      // Send request
      final streamedResponse = await request.send();
      final response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        return PredictionResponse.fromJson(data);
      } else {
        final error = json.decode(response.body);
        throw Exception(error['error'] ?? 'Failed to get prediction');
      }
    } catch (e) {
      throw Exception('Connection error: $e');
    }
  }

  /// Check if backend is accessible
  static Future<bool> checkConnection() async {
    try {
      final response = await http
          .get(Uri.parse(baseUrl))
          .timeout(const Duration(seconds: 5));
      return response.statusCode == 200;
    } catch (e) {
      return false;
    }
  }
}

class PredictionResponse {
  final bool success;
  final bool hasPneumonia;
  final double confidence;
  final Map<String, double> probabilities;
  final Map<String, double> features;
  final String processingTime;

  PredictionResponse({
    required this.success,
    required this.hasPneumonia,
    required this.confidence,
    required this.probabilities,
    required this.features,
    required this.processingTime,
  });

  factory PredictionResponse.fromJson(Map<String, dynamic> json) {
    return PredictionResponse(
      success: json['success'] ?? false,
      hasPneumonia: json['has_pneumonia'] ?? false,
      confidence: (json['confidence'] ?? 0).toDouble(),
      probabilities: {
        'normal': (json['probabilities']['NORMAL'] ?? 0).toDouble(),
        'pneumonia': (json['probabilities']['PNEUMONIA'] ?? 0).toDouble(),
      },
      features: Map<String, double>.from(
        json['features'].map((key, value) => MapEntry(key, value.toDouble())),
      ),
      processingTime: '${json['processing_time'] ?? 0}s',
    );
  }

  String get prediction => hasPneumonia ? 'Pneumonia Detected' : 'Normal';
}
