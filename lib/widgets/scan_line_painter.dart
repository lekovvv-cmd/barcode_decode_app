import 'package:flutter/material.dart';

class ScanLinePainter extends CustomPainter {
  final double progress; // 0..1
  final Rect roi;

  ScanLinePainter({
    required this.progress,
    required this.roi,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final rect = Rect.fromLTWH(
      roi.left * size.width,
      roi.top * size.height,
      roi.width * size.width,
      roi.height * size.height,
    );

    final y = rect.top + rect.height * progress;

    final paint = Paint()
      ..color = Colors.greenAccent.withOpacity(0.9)
      ..strokeWidth = 2.0;

    canvas.drawLine(Offset(rect.left, y), Offset(rect.right, y), paint);
  }

  @override
  bool shouldRepaint(covariant ScanLinePainter oldDelegate) {
    return oldDelegate.progress != progress || oldDelegate.roi != roi;
  }
}

