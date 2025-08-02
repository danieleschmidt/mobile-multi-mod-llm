# Test Fixtures

This directory contains static test data and fixtures used across the test suite.

## Directory Structure

```
fixtures/
├── images/           # Sample images for multimodal testing
│   ├── samples/      # Basic test images
│   ├── edge_cases/   # Edge case images (very small, large, corrupt)
│   └── adversarial/  # Adversarial examples for security testing
├── text/             # Text samples and datasets
│   ├── captions/     # Image caption datasets
│   ├── questions/    # VQA question datasets
│   └── documents/    # OCR test documents
├── models/           # Mock model artifacts
│   ├── checkpoints/  # Sample model checkpoints
│   ├── configs/      # Model configuration files
│   └── exports/      # Sample exported models (TFLite, CoreML, etc.)
├── datasets/         # Mock datasets for testing
│   ├── small/        # Small datasets for unit tests
│   ├── medium/       # Medium datasets for integration tests
│   └── large/        # Large datasets for performance tests
└── hardware/         # Hardware-specific test data
    ├── profiles/     # Performance profiles for different devices
    └── constraints/  # Hardware constraint configurations
```

## Usage Guidelines

### Fixture Naming Convention
- Use descriptive names that indicate the fixture purpose
- Include size indicators: `small_`, `medium_`, `large_`
- Include format indicators: `_jpg`, `_png`, `_tflite`, etc.
- Include quality indicators: `_high_res`, `_low_res`, `_corrupted`

### File Size Limits
- Unit test fixtures: < 1MB each
- Integration test fixtures: < 10MB each
- Performance test fixtures: < 100MB each
- Use compression when possible

### Creating New Fixtures

1. **Add to appropriate subdirectory**
2. **Update this README with description**
3. **Add to .gitignore if > 50MB**
4. **Create generator script if needed**

### Security Considerations
- No real user data in fixtures
- No sensitive information (API keys, credentials)
- Synthetic data only for privacy protection
- Adversarial examples should be clearly marked

## Fixture Catalog

### Images

#### Basic Samples (`images/samples/`)
- `rgb_224x224.jpg` - Standard RGB image (224x224)
- `grayscale_224x224.jpg` - Grayscale image
- `rgba_with_transparency.png` - Image with alpha channel
- `high_contrast.jpg` - High contrast image for OCR testing

#### Edge Cases (`images/edge_cases/`)
- `tiny_8x8.jpg` - Very small image (8x8 pixels)
- `large_4096x4096.jpg` - Large image (4096x4096 pixels)
- `corrupted_header.jpg` - Image with corrupted header
- `empty_file.jpg` - Empty file with .jpg extension
- `non_square_1920x1080.jpg` - Non-square aspect ratio

#### Adversarial (`images/adversarial/`)
- `fgsm_attack.jpg` - FGSM adversarial example
- `pgd_attack.jpg` - PGD adversarial example
- `steganography_hidden.png` - Image with hidden data

### Text

#### Captions (`text/captions/`)
- `coco_samples.json` - Sample COCO-style captions
- `multilingual_captions.json` - Multi-language captions
- `long_captions.json` - Very long caption examples

#### Questions (`text/questions/`)
- `vqa_samples.json` - Sample VQA questions and answers
- `yes_no_questions.json` - Binary yes/no questions
- `counting_questions.json` - Object counting questions

#### Documents (`text/documents/`)
- `ocr_test_document.pdf` - Multi-page document for OCR
- `handwritten_text.jpg` - Handwritten text samples
- `mathematical_equations.png` - Math formulas and equations

### Models

#### Checkpoints (`models/checkpoints/`)
- `tiny_model_state.pth` - Minimal model state for testing
- `quantized_int2_sample.pth` - INT2 quantized model sample
- `corrupted_checkpoint.pth` - Corrupted checkpoint for error testing

#### Configs (`models/configs/`)
- `minimal_config.yaml` - Minimal model configuration
- `full_config.yaml` - Complete configuration example
- `invalid_config.yaml` - Invalid configuration for error testing

#### Exports (`models/exports/`)
- `sample_model.tflite` - Sample TensorFlow Lite model
- `sample_model.mlmodel` - Sample Core ML model
- `sample_model.onnx` - Sample ONNX model

### Datasets

#### Small (`datasets/small/`)
- `mini_coco.json` - 10 samples from COCO-style dataset
- `mini_vqa.json` - 10 VQA question-answer pairs
- `mini_ocr.json` - 10 OCR text samples

#### Medium (`datasets/medium/`)
- `subset_coco.json` - 1000 samples for integration testing
- `subset_vqa.json` - 1000 VQA samples
- `subset_ocr.json` - 1000 OCR samples

#### Large (`datasets/large/`)
- `performance_dataset.json` - Large dataset for performance testing
- `stress_test_dataset.json` - Dataset for stress testing

### Hardware

#### Profiles (`hardware/profiles/`)
- `snapdragon_8gen3.json` - Performance profile for Snapdragon 8 Gen 3
- `apple_a17_pro.json` - Performance profile for Apple A17 Pro
- `generic_mobile.json` - Generic mobile device profile

#### Constraints (`hardware/constraints/`)
- `memory_limited.json` - Memory-constrained environment
- `cpu_only.json` - CPU-only inference constraints
- `battery_optimized.json` - Battery-optimized settings

## Maintenance

### Regular Tasks
- Review fixture sizes quarterly
- Update compressed archives when needed
- Validate fixture integrity
- Remove obsolete fixtures

### Automated Checks
- Pre-commit hooks validate fixture sizes
- CI checks fixture integrity
- Automated compression for large files

### Version Control
- Large files (>50MB) should use Git LFS
- Binary files should be compressed when possible
- Document any external dependencies

---

**Last Updated**: 2025-01-20  
**Maintainer**: Testing Team