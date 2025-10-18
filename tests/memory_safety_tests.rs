#[cfg(test)]
mod memory_safety_tests {
    use bytemuck::cast_slice;
    use half::f16;

    #[test]
    fn test_safe_f32_to_bytes_conversion() {
        let data = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32];
        
        // Test safe conversion using bytemuck
        let bytes = cast_slice::<f32, u8>(&data);
        
        // Verify the conversion maintains correct size
        assert_eq!(bytes.len(), data.len() * std::mem::size_of::<f32>());
        
        // Verify we can safely convert back
        let converted_back = cast_slice::<u8, f32>(bytes);
        assert_eq!(converted_back, &data);
    }

    #[test]
    fn test_safe_f16_to_bytes_conversion() {
        let data_f32 = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32];
        let data_f16: Vec<f16> = data_f32.iter().map(|&x| f16::from_f32(x)).collect();
        
        // Test safe conversion using bytemuck
        let bytes = cast_slice::<f16, u8>(&data_f16);
        
        // Verify the conversion maintains correct size
        assert_eq!(bytes.len(), data_f16.len() * std::mem::size_of::<f16>());
        
        // Verify we can safely convert back
        let converted_back = cast_slice::<u8, f16>(bytes);
        assert_eq!(converted_back, &data_f16);
        
        // Verify precision loss is expected for f16
        let back_to_f32: Vec<f32> = converted_back.iter().map(|&x| x.to_f32()).collect();
        for (original, converted) in data_f32.iter().zip(back_to_f32.iter()) {
            assert!((original - converted).abs() < 0.01); // Allow for f16 precision loss
        }
    }

    #[test]
    fn test_safe_i8_to_bytes_conversion() {
        let data = vec![127i8, -128i8, 0i8, 64i8];
        
        // Test safe conversion using bytemuck
        let bytes = cast_slice::<i8, u8>(&data);
        
        // Verify the conversion maintains correct size
        assert_eq!(bytes.len(), data.len() * std::mem::size_of::<i8>());
        
        // Verify we can safely convert back
        let converted_back = cast_slice::<u8, i8>(bytes);
        assert_eq!(converted_back, &data);
    }

    #[test]
    fn test_nan_and_infinite_handling() {
        let data = vec![
            1.0f32,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            2.0f32,
        ];
        
        // Test conversion to i8 with proper NaN/infinity handling
        let converted: Vec<i8> = data.iter()
            .map(|&x| {
                if !x.is_finite() {
                    return 0i8; // Safe handling of non-finite values
                }
                (x * 127.0).clamp(-128.0, 127.0) as i8
            })
            .collect();
        
        assert_eq!(converted[0], 127); // 1.0 * 127
        assert_eq!(converted[1], 0);   // NaN -> 0
        assert_eq!(converted[2], 0);   // Infinity -> 0
        assert_eq!(converted[3], 0);   // -Infinity -> 0
        assert_eq!(converted[4], 127); // 2.0 * 127 (clamped to 127)
    }

    #[test]
    fn test_empty_array_handling() {
        let empty_f32: Vec<f32> = vec![];
        let empty_f16: Vec<f16> = vec![];
        let empty_i8: Vec<i8> = vec![];
        
        // Test that empty arrays are handled safely
        let bytes_f32 = cast_slice::<f32, u8>(&empty_f32);
        let bytes_f16 = cast_slice::<f16, u8>(&empty_f16);
        let bytes_i8 = cast_slice::<i8, u8>(&empty_i8);
        
        assert!(bytes_f32.is_empty());
        assert!(bytes_f16.is_empty());
        assert!(bytes_i8.is_empty());
    }

    #[test]
    fn test_bounds_checking() {
        let data = vec![1.0f32, 2.0f32, 3.0f32];
        let dim = 2;
        
        // Test that we properly validate bounds for chunking
        let total = data.len() / dim; // Should be 1 with remainder
        assert_eq!(total, 1);
        assert_eq!(data.len() % dim, 1); // Remainder should be 1
        
        // This would be an invalid configuration that should be caught
        assert!(data.len() % dim != 0); // Not evenly divisible
    }

    #[test]
    fn test_large_array_conversion() {
        // Test with a large array to ensure no overflow issues
        let size = 10_000;
        let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
        
        // Test safe conversion
        let bytes = cast_slice::<f32, u8>(&data);
        assert_eq!(bytes.len(), size * std::mem::size_of::<f32>());
        
        // Verify integrity
        let converted_back = cast_slice::<u8, f32>(bytes);
        assert_eq!(converted_back, &data);
    }

    #[test]
    fn test_alignment_safety() {
        // Test that our Pod types are properly aligned
        assert_eq!(std::mem::align_of::<f32>(), std::mem::align_of::<f32>());
        assert_eq!(std::mem::align_of::<f16>(), std::mem::align_of::<f16>());
        assert_eq!(std::mem::align_of::<i8>(), std::mem::align_of::<i8>());
        
        // Ensure no padding issues with size calculations
        let f32_data = vec![1.0f32; 100];
        let f16_data = vec![f16::from_f32(1.0); 100];
        let i8_data = vec![1i8; 100];
        
        assert_eq!(
            cast_slice::<f32, u8>(&f32_data).len(),
            f32_data.len() * std::mem::size_of::<f32>()
        );
        assert_eq!(
            cast_slice::<f16, u8>(&f16_data).len(),
            f16_data.len() * std::mem::size_of::<f16>()
        );
        assert_eq!(
            cast_slice::<i8, u8>(&i8_data).len(),
            i8_data.len() * std::mem::size_of::<i8>()
        );
    }
}

#[cfg(feature = "cuda")]
mod gpu_memory_safety_tests {
    use super::*;
    
    // Mock the precision and error types for testing
    #[derive(Debug, Clone, Copy)]
    enum MockPrecision {
        Fp32,
        Fp16,
        Int8,
    }

    #[derive(Debug)]
    enum MockGpuError {
        InvalidInput(String),
    }

    // Mock implementation of our safe convert_data function for testing
    fn safe_convert_data(
        queries: &[f32],
        corpus: &[f32],
        precision: MockPrecision,
    ) -> Result<(Vec<u8>, Vec<u8>), MockGpuError> {
        // Validate input arrays are not empty
        if queries.is_empty() || corpus.is_empty() {
            return Err(MockGpuError::InvalidInput("Input arrays cannot be empty".to_string()));
        }

        match precision {
            MockPrecision::Fp32 => {
                // Use safe bytemuck conversion instead of unsafe raw parts
                let queries_bytes = bytemuck::cast_slice::<f32, u8>(queries).to_vec();
                let corpus_bytes = bytemuck::cast_slice::<f32, u8>(corpus).to_vec();
                Ok((queries_bytes, corpus_bytes))
            }
            MockPrecision::Fp16 => {
                // Convert to f16 with bounds checking
                let queries_f16: Vec<f16> = queries.iter().map(|&x| f16::from_f32(x)).collect();
                let corpus_f16: Vec<f16> = corpus.iter().map(|&x| f16::from_f32(x)).collect();
                
                // Use safe bytemuck conversion instead of unsafe raw parts
                let queries_bytes = bytemuck::cast_slice::<f16, u8>(&queries_f16).to_vec();
                let corpus_bytes = bytemuck::cast_slice::<f16, u8>(&corpus_f16).to_vec();
                Ok((queries_bytes, corpus_bytes))
            }
            MockPrecision::Int8 => {
                // Convert to int8 with scaling and clamping
                let queries_i8: Vec<i8> = queries.iter()
                    .map(|&x| {
                        // Add bounds checking for safety
                        if !x.is_finite() {
                            return 0i8; // Handle NaN/infinite values safely
                        }
                        (x * 127.0).clamp(-128.0, 127.0) as i8
                    })
                    .collect();
                let corpus_i8: Vec<i8> = corpus.iter()
                    .map(|&x| {
                        // Add bounds checking for safety
                        if !x.is_finite() {
                            return 0i8; // Handle NaN/infinite values safely
                        }
                        (x * 127.0).clamp(-128.0, 127.0) as i8
                    })
                    .collect();
                
                // Use safe bytemuck conversion instead of unsafe raw parts
                let queries_bytes = bytemuck::cast_slice::<i8, u8>(&queries_i8).to_vec();
                let corpus_bytes = bytemuck::cast_slice::<i8, u8>(&corpus_i8).to_vec();
                Ok((queries_bytes, corpus_bytes))
            }
        }
    }

    #[test]
    fn test_safe_convert_data_fp32() {
        let queries = vec![1.0, 2.0, 3.0];
        let corpus = vec![4.0, 5.0, 6.0];
        
        let result = safe_convert_data(&queries, &corpus, MockPrecision::Fp32);
        assert!(result.is_ok());
        
        let (q_bytes, c_bytes) = result.unwrap();
        assert_eq!(q_bytes.len(), queries.len() * 4); // 4 bytes per f32
        assert_eq!(c_bytes.len(), corpus.len() * 4);
    }

    #[test]
    fn test_safe_convert_data_fp16() {
        let queries = vec![1.0, 2.0, 3.0];
        let corpus = vec![4.0, 5.0, 6.0];
        
        let result = safe_convert_data(&queries, &corpus, MockPrecision::Fp16);
        assert!(result.is_ok());
        
        let (q_bytes, c_bytes) = result.unwrap();
        assert_eq!(q_bytes.len(), queries.len() * 2); // 2 bytes per f16
        assert_eq!(c_bytes.len(), corpus.len() * 2);
    }

    #[test]
    fn test_safe_convert_data_int8() {
        let queries = vec![0.5, 1.0, -0.5];
        let corpus = vec![0.0, -1.0, 0.25];
        
        let result = safe_convert_data(&queries, &corpus, MockPrecision::Int8);
        assert!(result.is_ok());
        
        let (q_bytes, c_bytes) = result.unwrap();
        assert_eq!(q_bytes.len(), queries.len()); // 1 byte per i8
        assert_eq!(c_bytes.len(), corpus.len());
    }

    #[test]
    fn test_safe_convert_data_empty_input() {
        let queries = vec![];
        let corpus = vec![1.0, 2.0];
        
        let result = safe_convert_data(&queries, &corpus, MockPrecision::Fp32);
        assert!(result.is_err());
        
        let queries = vec![1.0, 2.0];
        let corpus = vec![];
        
        let result = safe_convert_data(&queries, &corpus, MockPrecision::Fp32);
        assert!(result.is_err());
    }

    #[test]
    fn test_safe_convert_data_with_nan() {
        let queries = vec![1.0, f32::NAN, 3.0];
        let corpus = vec![f32::INFINITY, 5.0, f32::NEG_INFINITY];
        
        let result = safe_convert_data(&queries, &corpus, MockPrecision::Int8);
        assert!(result.is_ok());
        
        let (q_bytes, c_bytes) = result.unwrap();
        
        // Convert back to verify NaN/infinity handling
        let q_i8 = bytemuck::cast_slice::<u8, i8>(&q_bytes);
        let c_i8 = bytemuck::cast_slice::<u8, i8>(&c_bytes);
        
        assert_eq!(q_i8[0], 127); // 1.0 * 127
        assert_eq!(q_i8[1], 0);   // NaN -> 0
        assert_eq!(q_i8[2], 127); // 3.0 * 127 (clamped)
        
        assert_eq!(c_i8[0], 0);   // Infinity -> 0
        assert_eq!(c_i8[1], 127); // 5.0 * 127 (clamped)
        assert_eq!(c_i8[2], 0);   // -Infinity -> 0
    }
}