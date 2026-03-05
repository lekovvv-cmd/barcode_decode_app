import 'package:camera/camera.dart';

class CameraService {
  Future<CameraController> initializeCamera() async {
    final cameras = await availableCameras();

    final backCamera = cameras.firstWhere(
      (camera) => camera.lensDirection == CameraLensDirection.back,
      orElse: () => cameras.first,
    );

    final controller = CameraController(
      backCamera,
      ResolutionPreset.medium,
      enableAudio: false,
    );

    await controller.initialize();
    return controller;
  }
}

