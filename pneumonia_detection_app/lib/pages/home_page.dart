import 'dart:convert';
import 'dart:html' as html;
import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import '../app_theme.dart';
import '../services/api_service.dart';

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage>
    with SingleTickerProviderStateMixin {
  PlatformFile? _selectedFile;
  bool _isProcessing = false;
  PredictionResult? _result;
  late AnimationController _animationController;
  late Animation<double> _fadeAnimation;

  @override
  void initState() {
    super.initState();
    _animationController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 500),
    );
    _fadeAnimation = CurvedAnimation(
      parent: _animationController,
      curve: Curves.easeInOut,
    );
  }

  @override
  void dispose() {
    _animationController.dispose();
    super.dispose();
  }

  Future<void> _pickFile() async {
    try {
      FilePickerResult? result = await FilePicker.platform.pickFiles(
        type: FileType.image,
        allowMultiple: false,
      );

      if (result != null) {
        setState(() {
          _selectedFile = result.files.first;
          _result = null; // Clear previous results
        });
        _animationController.forward(from: 0);
      }
    } catch (e) {
      _showErrorSnackBar('Error picking file: $e');
    }
  }

  Future<void> _predictPneumonia() async {
    if (_selectedFile == null) return;

    setState(() {
      _isProcessing = true;
    });

    try {
      // Call real API
      final response = await ApiService.predictPneumonia(_selectedFile!);

      // Convert response to PredictionResult
      final result = PredictionResult(
        prediction: response.prediction,
        confidence: response.confidence / 100, // Convert to decimal
        probabilities: response.probabilities,
        timeTaken: response.processingTime,
        features: response.features,
      );

      setState(() {
        _result = result;
        _isProcessing = false;
      });

      _animationController.forward(from: 0);
    } catch (e) {
      setState(() {
        _isProcessing = false;
      });
      _showErrorSnackBar(
        'Prediction failed: ${e.toString().replaceAll('Exception: ', '')}',
      );
    }
  }

  void _downloadFeatures() {
    if (_result == null) return;

    final jsonString = const JsonEncoder.withIndent(
      '  ',
    ).convert(_result!.features);
    final bytes = utf8.encode(jsonString);
    final blob = html.Blob([bytes]);
    final url = html.Url.createObjectUrlFromBlob(blob);
    html.AnchorElement(href: url)
      ..setAttribute('download', 'pneumonia_features.json')
      ..click();
    html.Url.revokeObjectUrl(url);

    _showSuccessSnackBar('Features downloaded successfully');
  }

  void _copyFeatures() {
    if (_result == null) return;

    final jsonString = const JsonEncoder.withIndent(
      '  ',
    ).convert(_result!.features);

    html.window.navigator.clipboard?.writeText(jsonString);
    _showSuccessSnackBar('Features copied to clipboard');
  }

  void _showErrorSnackBar(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        backgroundColor: Theme.of(context).colorScheme.error,
        behavior: SnackBarBehavior.floating,
      ),
    );
  }

  void _showSuccessSnackBar(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        backgroundColor: AppTheme.successLight,
        behavior: SnackBarBehavior.floating,
      ),
    );
  }

  void _reset() {
    setState(() {
      _selectedFile = null;
      _result = null;
      _isProcessing = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final isDark = theme.brightness == Brightness.dark;

    return Scaffold(
      backgroundColor: theme.scaffoldBackgroundColor,
      body: Center(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 24.0, vertical: 16.0),
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 1200),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                // Header
                Padding(
                  padding: const EdgeInsets.only(bottom: 16),
                  child: Column(
                    children: [
                      Icon(
                        Icons.medical_services_rounded,
                        size: 48,
                        color: theme.colorScheme.primary,
                      ),
                      const SizedBox(height: 8),
                      Text(
                        'Pneumonia Detection System',
                        style: theme.textTheme.headlineMedium?.copyWith(
                          fontWeight: FontWeight.bold,
                          color: theme.colorScheme.onSurface,
                        ),
                        textAlign: TextAlign.center,
                      ),
                      const SizedBox(height: 4),
                      Text(
                        'Upload a chest X-ray image for AI-powered pneumonia detection',
                        style: theme.textTheme.bodyMedium?.copyWith(
                          color: theme.colorScheme.onSurfaceVariant,
                        ),
                        textAlign: TextAlign.center,
                      ),
                    ],
                  ),
                ),

                // Main Content
                Expanded(
                  child: Row(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      // Left Panel - Upload & Preview
                      Expanded(
                        flex: 5,
                        child: Card(
                          elevation: isDark ? 4 : 3,
                          shadowColor: theme.colorScheme.shadow.withValues(
                            alpha: 0.3,
                          ),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(16),
                            side: BorderSide(
                              color: theme.colorScheme.outline.withValues(
                                alpha: 0.2,
                              ),
                              width: 1,
                            ),
                          ),
                          child: Padding(
                            padding: const EdgeInsets.all(24),
                            child: Column(
                              mainAxisSize: MainAxisSize.min,
                              children: [
                                // Upload Area
                                Flexible(child: _buildUploadArea(theme)),

                                if (_selectedFile != null) ...[
                                  const SizedBox(height: 24),
                                  _buildSelectedFileInfo(theme),
                                  const SizedBox(height: 24),
                                  _buildActionButtons(theme),
                                ],
                              ],
                            ),
                          ),
                        ),
                      ),

                      const SizedBox(width: 24),

                      // Right Panel - Results
                      Expanded(
                        flex: 5,
                        child: _result != null
                            ? FadeTransition(
                                opacity: _fadeAnimation,
                                child: _buildResultsCard(theme),
                              )
                            : _buildPlaceholderCard(theme),
                      ),
                    ],
                  ),
                ),

                // Footer
                Padding(
                  padding: const EdgeInsets.only(top: 12),
                  child: Text(
                    '© 2026 Pneumonia Detection System. For research purposes only.',
                    style: theme.textTheme.bodySmall?.copyWith(
                      color: theme.colorScheme.onSurfaceVariant,
                    ),
                    textAlign: TextAlign.center,
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildUploadArea(ThemeData theme) {
    return InkWell(
      onTap: _isProcessing ? null : _pickFile,
      borderRadius: BorderRadius.circular(12),
      child: Container(
        constraints: const BoxConstraints(minHeight: 280, maxHeight: 280),
        decoration: BoxDecoration(
          border: Border.all(
            color: theme.colorScheme.outline,
            width: 2,
            strokeAlign: BorderSide.strokeAlignInside,
          ),
          borderRadius: BorderRadius.circular(12),
          color: theme.colorScheme.surfaceContainerHighest.withValues(
            alpha: 0.3,
          ),
        ),
        child: _selectedFile != null && _selectedFile!.bytes != null
            ? Padding(
                padding: const EdgeInsets.all(12.0),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(10),
                  child: Image.memory(
                    _selectedFile!.bytes!,
                    fit: BoxFit.contain,
                  ),
                ),
              )
            : Center(
                child: Padding(
                  padding: const EdgeInsets.all(20.0),
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(
                        Icons.cloud_upload_outlined,
                        size: 64,
                        color: theme.colorScheme.primary,
                      ),
                      const SizedBox(height: 16),
                      Text(
                        'Click to upload X-ray image',
                        style: theme.textTheme.titleMedium?.copyWith(
                          color: theme.colorScheme.onSurface,
                          fontWeight: FontWeight.w600,
                        ),
                        textAlign: TextAlign.center,
                      ),
                      const SizedBox(height: 8),
                      Text(
                        'Supports: JPG, PNG, DICOM',
                        style: theme.textTheme.bodyMedium?.copyWith(
                          color: theme.colorScheme.onSurfaceVariant,
                        ),
                        textAlign: TextAlign.center,
                      ),
                    ],
                  ),
                ),
              ),
      ),
    );
  }

  Widget _buildSelectedFileInfo(ThemeData theme) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: theme.colorScheme.primaryContainer,
        borderRadius: BorderRadius.circular(12),
      ),
      child: Row(
        children: [
          Icon(
            Icons.image_outlined,
            color: theme.colorScheme.onPrimaryContainer,
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  _selectedFile!.name,
                  style: theme.textTheme.bodyLarge?.copyWith(
                    color: theme.colorScheme.onPrimaryContainer,
                    fontWeight: FontWeight.w600,
                  ),
                  overflow: TextOverflow.ellipsis,
                ),
                const SizedBox(height: 4),
                Text(
                  '${(_selectedFile!.size / 1024).toStringAsFixed(2)} KB',
                  style: theme.textTheme.bodySmall?.copyWith(
                    color: theme.colorScheme.onPrimaryContainer.withValues(
                      alpha: 0.8,
                    ),
                  ),
                ),
              ],
            ),
          ),
          IconButton(
            onPressed: _reset,
            icon: Icon(
              Icons.close,
              color: theme.colorScheme.onPrimaryContainer,
            ),
            tooltip: 'Remove image',
          ),
        ],
      ),
    );
  }

  Widget _buildActionButtons(ThemeData theme) {
    return Row(
      children: [
        Expanded(
          child: FilledButton.icon(
            onPressed: _isProcessing ? null : _predictPneumonia,
            icon: _isProcessing
                ? SizedBox(
                    width: 20,
                    height: 20,
                    child: CircularProgressIndicator(
                      strokeWidth: 2,
                      valueColor: AlwaysStoppedAnimation<Color>(
                        theme.colorScheme.onPrimary,
                      ),
                    ),
                  )
                : const Icon(Icons.analytics_outlined),
            label: Text(_isProcessing ? 'Analyzing...' : 'Predict'),
            style: FilledButton.styleFrom(
              padding: const EdgeInsets.symmetric(vertical: 16),
              textStyle: theme.textTheme.titleMedium,
            ),
          ),
        ),
        const SizedBox(width: 12),
        OutlinedButton.icon(
          onPressed: _isProcessing ? null : _reset,
          icon: const Icon(Icons.refresh),
          label: const Text('Reset'),
          style: OutlinedButton.styleFrom(
            padding: const EdgeInsets.symmetric(vertical: 16, horizontal: 24),
            textStyle: theme.textTheme.titleMedium,
          ),
        ),
      ],
    );
  }

  Widget _buildPlaceholderCard(ThemeData theme) {
    return Card(
      elevation: theme.brightness == Brightness.dark ? 4 : 3,
      shadowColor: theme.colorScheme.shadow.withValues(alpha: 0.3),
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16),
        side: BorderSide(
          color: theme.colorScheme.outline.withValues(alpha: 0.2),
          width: 1,
        ),
      ),
      child: Container(
        height: double.infinity,
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              Icons.insert_chart_outlined,
              size: 64,
              color: theme.colorScheme.onSurfaceVariant.withValues(alpha: 0.3),
            ),
            const SizedBox(height: 16),
            Text(
              'Results will appear here',
              style: theme.textTheme.titleLarge?.copyWith(
                color: theme.colorScheme.onSurfaceVariant.withValues(
                  alpha: 0.6,
                ),
              ),
            ),
            const SizedBox(height: 8),
            Text(
              'Upload an X-ray image and click Predict',
              style: theme.textTheme.bodyMedium?.copyWith(
                color: theme.colorScheme.onSurfaceVariant.withValues(
                  alpha: 0.5,
                ),
              ),
              textAlign: TextAlign.center,
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildResultsCard(ThemeData theme) {
    final result = _result!;
    final isPneumonia = result.prediction.toLowerCase().contains('pneumonia');

    return Card(
      elevation: theme.brightness == Brightness.dark ? 4 : 3,
      shadowColor: theme.colorScheme.shadow.withValues(alpha: 0.3),
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16),
        side: BorderSide(
          color: theme.colorScheme.outline.withValues(alpha: 0.2),
          width: 1,
        ),
      ),
      child: Container(
        height: double.infinity,
        child: SingleChildScrollView(
          child: Padding(
            padding: const EdgeInsets.all(24),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // Result Header
                Row(
                  children: [
                    Icon(
                      isPneumonia
                          ? Icons.warning_amber_rounded
                          : Icons.check_circle_outline,
                      size: 32,
                      color: isPneumonia
                          ? AppTheme.warningLight
                          : AppTheme.successLight,
                    ),
                    const SizedBox(width: 12),
                    Text(
                      'Detection Result',
                      style: theme.textTheme.headlineSmall?.copyWith(
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 24),

                // Prediction
                _buildResultItem(
                  theme,
                  'Prediction',
                  result.prediction,
                  icon: Icons.info_outline,
                  valueColor: isPneumonia
                      ? AppTheme.warningLight
                      : AppTheme.successLight,
                ),
                const SizedBox(height: 16),

                // Confidence
                _buildResultItem(
                  theme,
                  'Confidence',
                  '${(result.confidence * 100).toStringAsFixed(2)}%',
                  icon: Icons.speed,
                ),
                const SizedBox(height: 16),

                // Confidence Bar
                _buildConfidenceBar(theme, result.confidence),
                const SizedBox(height: 24),

                // Probabilities Section
                Container(
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    color: theme.colorScheme.surfaceContainerHighest.withValues(
                      alpha: 0.3,
                    ),
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(
                      color: theme.colorScheme.outline.withValues(alpha: 0.2),
                    ),
                  ),
                  child: Column(
                    children: [
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          Row(
                            children: [
                              Icon(
                                Icons.health_and_safety_outlined,
                                size: 18,
                                color: AppTheme.successLight,
                              ),
                              const SizedBox(width: 8),
                              Text(
                                'Normal',
                                style: theme.textTheme.bodyMedium?.copyWith(
                                  fontWeight: FontWeight.w500,
                                ),
                              ),
                            ],
                          ),
                          Text(
                            '${result.probabilities['normal']!.toStringAsFixed(2)}%',
                            style: theme.textTheme.bodyMedium?.copyWith(
                              fontWeight: FontWeight.bold,
                              color: AppTheme.successLight,
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 12),
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          Row(
                            children: [
                              Icon(
                                Icons.coronavirus_outlined,
                                size: 18,
                                color: AppTheme.warningLight,
                              ),
                              const SizedBox(width: 8),
                              Text(
                                'Pneumonia',
                                style: theme.textTheme.bodyMedium?.copyWith(
                                  fontWeight: FontWeight.w500,
                                ),
                              ),
                            ],
                          ),
                          Text(
                            '${result.probabilities['pneumonia']!.toStringAsFixed(2)}%',
                            style: theme.textTheme.bodyMedium?.copyWith(
                              fontWeight: FontWeight.bold,
                              color: AppTheme.warningLight,
                            ),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 16),

                // Time Taken
                _buildResultItem(
                  theme,
                  'Processing Time',
                  result.timeTaken,
                  icon: Icons.timer_outlined,
                ),
                const SizedBox(height: 24),

                const Divider(),
                const SizedBox(height: 24),

                // Features Section
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Text(
                      'Extracted Features (${result.features.length})',
                      style: theme.textTheme.titleMedium?.copyWith(
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                    Row(
                      children: [
                        TextButton.icon(
                          onPressed: _copyFeatures,
                          icon: const Icon(Icons.copy, size: 18),
                          label: const Text('Copy'),
                        ),
                        const SizedBox(width: 8),
                        FilledButton.icon(
                          onPressed: _downloadFeatures,
                          icon: const Icon(Icons.download, size: 18),
                          label: const Text('Download'),
                        ),
                      ],
                    ),
                  ],
                ),
                const SizedBox(height: 12),
                Text(
                  'All ${result.features.length} feature values are available for copy/download',
                  style: theme.textTheme.bodySmall?.copyWith(
                    color: theme.colorScheme.onSurfaceVariant,
                    fontStyle: FontStyle.italic,
                  ),
                ),
                const SizedBox(height: 16),

                // No feature preview - just buttons above
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildResultItem(
    ThemeData theme,
    String label,
    String value, {
    IconData? icon,
    Color? valueColor,
  }) {
    return Row(
      children: [
        if (icon != null) ...[
          Icon(icon, size: 20, color: theme.colorScheme.onSurfaceVariant),
          const SizedBox(width: 8),
        ],
        Text(
          '$label: ',
          style: theme.textTheme.bodyLarge?.copyWith(
            color: theme.colorScheme.onSurfaceVariant,
          ),
        ),
        Text(
          value,
          style: theme.textTheme.bodyLarge?.copyWith(
            fontWeight: FontWeight.w600,
            color: valueColor ?? theme.colorScheme.onSurface,
          ),
        ),
      ],
    );
  }

  Widget _buildConfidenceBar(ThemeData theme, double confidence) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        ClipRRect(
          borderRadius: BorderRadius.circular(8),
          child: LinearProgressIndicator(
            value: confidence,
            minHeight: 12,
            backgroundColor: theme.colorScheme.surfaceContainerHighest,
            valueColor: AlwaysStoppedAnimation<Color>(
              confidence > 0.7 ? AppTheme.successLight : AppTheme.warningLight,
            ),
          ),
        ),
        const SizedBox(height: 4),
        Text(
          confidence > 0.7 ? 'High Confidence' : 'Moderate Confidence',
          style: theme.textTheme.bodySmall?.copyWith(
            color: theme.colorScheme.onSurfaceVariant,
            fontStyle: FontStyle.italic,
          ),
        ),
      ],
    );
  }
}

class PredictionResult {
  final String prediction;
  final double confidence;
  final Map<String, double> probabilities;
  final String timeTaken;
  final Map<String, double> features;

  PredictionResult({
    required this.prediction,
    required this.confidence,
    required this.probabilities,
    required this.timeTaken,
    required this.features,
  });
}
