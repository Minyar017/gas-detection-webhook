import 'dart:async';
import 'dart:convert';
import 'package:http/http.dart' as http;

class AlertService {
  static const String _baseUrl = 'https://alert-u1o7.onrender.com/predict'; // Use Render-hosted API
  static const int _timeoutSeconds = 15;
  static const int _maxRetries = 3;
  static const Duration _retryDelay = Duration(seconds: 2);

  double? _lastTemperature;
  double? _lastHumidity;
  double? _lastMq5;
  double? _lastMq7;

  bool _hasChanged(double temperature, double humidity, double mq5, double mq7) {
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
      return {'alerts': [], 'danger': false, 'suspected_gas': 'Unknown'}; // Pas de changement, pas de nouvelle requête
    }

    int retryCount = 0;
    while (retryCount < _maxRetries) {
      try {
        print(
          "Sending request to Flask (Attempt $retryCount): temp=$temperature, humidity=$humidity, mq5=$mq5, mq7=$mq7",
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
            .timeout(Duration(seconds: _timeoutSeconds));

        if (response.statusCode == 200) {
          print("Flask response received: ${response.body}");
          final result = jsonDecode(response.body) as Map<String, dynamic>;
          _updateLastValues(temperature, humidity, mq5, mq7);
          return result;
        } else {
          print(
            "Flask error: Status code ${response.statusCode}, Body: ${response.body}",
          );
          retryCount++;
          if (retryCount < _maxRetries) {
            await Future.delayed(_retryDelay);
            continue;
          }
          return {'alerts': [], 'danger': false, 'suspected_gas': 'Unknown'};
        }
      } catch (e) {
        print("Error calling Flask (Attempt $retryCount): $e");
        retryCount++;
        if (retryCount < _maxRetries) {
          await Future.delayed(_retryDelay);
          continue;
        }
        return {'alerts': [], 'danger': false, 'suspected_gas': 'Unknown'};
      }
    }
    _updateLastValues(temperature, humidity, mq5, mq7);
    return {'alerts': [], 'danger': false, 'suspected_gas': 'Unknown'};
  }

  void _updateLastValues(double temperature, double humidity, double mq5, double mq7) {
    _lastTemperature = temperature;
    _lastHumidity = humidity;
    _lastMq5 = mq5;
    _lastMq7 = mq7;
  }
}
