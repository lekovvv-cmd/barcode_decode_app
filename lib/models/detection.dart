class Detection {
  final double x1;
  final double y1;
  final double x2;
  final double y2;
  final double confidence;
  final String decoded;

  Detection({
    required this.x1,
    required this.y1,
    required this.x2,
    required this.y2,
    required this.confidence,
    required this.decoded,
  });

  factory Detection.fromJson(Map<String, dynamic> json) {
    final bbox =
        (json['bbox'] as List).map((e) => (e as num).toDouble()).toList();
    return Detection(
      x1: bbox[0],
      y1: bbox[1],
      x2: bbox[2],
      y2: bbox[3],
      confidence: (json['confidence'] as num).toDouble(),
      decoded: json['decoded'] as String,
    );
  }

  static List<Detection> listFromJson(Map<String, dynamic> json) {
    final detectionsJson = json['detections'] as List<dynamic>? ?? [];
    return detectionsJson
        .map((item) => Detection.fromJson(item as Map<String, dynamic>))
        .toList();
  }
}

