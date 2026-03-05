import 'export_service_io.dart'
    if (dart.library.html) 'export_service_web.dart';

abstract class ExportService {
  Future<void> exportCsv(String csvContent);
}

ExportService createExportService() => getExportService();

