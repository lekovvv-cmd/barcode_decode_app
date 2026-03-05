import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

class TrackedBarcode {
  final int id;
  Rect bbox; // normalized bbox in [0,1]
  int framesSeen;
  int framesMissing;
  final List<XFile> frames;
  final Map<XFile, double> frameSharpness;
  double stabilityScore;

  TrackedBarcode({
    required this.id,
    required this.bbox,
    this.framesSeen = 1,
    this.framesMissing = 0,
    this.stabilityScore = 0,
    List<XFile>? frames,
    Map<XFile, double>? frameSharpness,
  })  : frames = frames ?? [],
        frameSharpness = frameSharpness ?? {};

  void addFrame(XFile frame, double sharpness, {int maxFrames = 30}) {
    frames.add(frame);
    frameSharpness[frame] = sharpness;
    if (frames.length > maxFrames) {
      final removed = frames.removeAt(0);
      frameSharpness.remove(removed);
    }
  }

  List<XFile> getBestFrames({int maxCount = 10}) {
    final sorted = [...frames]
      ..sort((a, b) =>
          (frameSharpness[b] ?? 0).compareTo(frameSharpness[a] ?? 0));
    return sorted.take(maxCount).toList();
  }

  bool get readyForFirstDecode => framesSeen >= 4;

  bool get readyForMultiFrameDecode => framesSeen >= 8;

  bool get readyForBestFrameDecode => framesSeen >= 15;
}


