import 'dart:async';
import 'dart:convert';
import 'package:http/http.dart' as http;

class AlertService {
  static const String _baseUrl =
      'https://alert-u1o7.onrender.com/predict'; // Use Render-hosted API
  static const int _timeoutSeconds = 15;
  static const int _maxRetries = 3;
  static const Duration _retryDelay = Duration(seconds: 2);

  double? _lastTemperature;
  double? _lastHumidity;
  double? _lastMq5;
  double? _lastMq7;

  bool _hasChanged(
    double temperature,
    double humidity,
    double mq5,
    double mq7,
  ) {
    return temperature != _lastTemperature ||
        humidity != _lastHumidity ||
        mq5 != _lastMq5 ||
        mq7 != _lastMq7;
  }

  Future<Map<String, dynamic>> checkForAlerts(
    double temperature,
    double humidity,
    double mq5,
    double mq7,
  ) async {
    if (!_hasChanged(temperature, humidity, mq5, mq7)) {
      print("No change in values. Skipping request.");
      return {
        'alerts': [],
        'danger': false,
        'prediction': 0,
        'alert_pred': 0,
        'suspected_gas': 'None',
        'mq5_pred': mq5,
        'mq7_pred': mq7,
      };
    }

    int attempt = 0;
    while (attempt < _maxRetries) {
      try {
        print(
          "Attempt ${attempt + 1}: Sending request to Flask: temp=$temperature, humidity=$humidity, mq5=$mq5, mq7=$mq7",
        );

        final response = await http
            .post(
              Uri.parse(_baseUrl),
              headers: {'Content-Type': 'application/json'},
              body: jsonEncode({
                'temperature': temperature,
                'humidity': humidity,
                'mq5': mq5,
                'mq7': mq7,
              }),
            )
            .timeout(const Duration(seconds: _timeoutSeconds));

        print("Response status: ${response.statusCode}");
        print("Response body: ${response.body}");

        if (response.statusCode == 200) {
          _lastTemperature = temperature;
          _lastHumidity = humidity;
          _lastMq5 = mq5;
          _lastMq7 = mq7;
          final responseData = jsonDecode(response.body);
          print(
            "Parsed response - alerts: ${responseData['alerts']}, danger: ${responseData['danger']}, suspected_gas: ${responseData['suspected_gas']}",
          );
          return responseData;
        } else {
          throw Exception(
            'Flask error: Status code ${response.statusCode}, Body: ${response.body}',
          );
        }
      } on http.ClientException catch (e) {
        print("Network error on attempt ${attempt + 1}: ${e.message}");
        if (attempt == _maxRetries - 1) {
          throw Exception(
            'Network error: Please check server connection and try again',
          );
        }
      } on TimeoutException catch (e) {
        print(
          "Request timed out on attempt ${attempt + 1} after $_timeoutSeconds seconds",
        );
        if (attempt == _maxRetries - 1) {
          throw Exception(
            'Server timeout: Please check your network connection',
          );
        }
      } catch (e, stackTrace) {
        print("Unexpected error on attempt ${attempt + 1}: $e");
        print("Stack trace: $stackTrace");
        if (attempt == _maxRetries - 1) {
          throw Exception('Unexpected error occurred: $e');
        }
      }
      attempt++;
      await Future.delayed(_retryDelay);
    }
    throw Exception('Failed to connect to server after $_maxRetries attempts');
  }
}
