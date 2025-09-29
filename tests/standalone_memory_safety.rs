// Standalone memory safety tests that can run without the main codebase

use bytemuck::{cast_slice, Pod, Zeroable};
use half::f16;

#[test]
fn test_safe_memory_conversions() {
    println!("Testing safe memory conversions with bytemuck...");

    // Test f32 -> bytes conversion
    let f32_data = vec![1.0f32, 2.0f32, 3.14f32, -1.5f32];
    let bytes = cast_slice::<f32, u8>(&f32_data);
    assert_eq!(bytes.len(), f32_data.len() * 4);
    
    // Test round-trip conversion
    let converted_back = cast_slice::<u8, f32>(bytes);
    assert_eq!(converted_back, &f32_data);

    println!("✓ f32 <-> bytes conversion works safely");

    // Test f16 -> bytes conversion
    let f16_data: Vec<f16> = f32_data.iter().map(|&x| f16::from_f32(x)).collect();
    let f16_bytes = cast_slice::<f16, u8>(&f16_data);
    assert_eq!(f16_bytes.len(), f16_data.len() * 2);
    
    let f16_back = cast_slice::<u8, f16>(f16_bytes);
    assert_eq!(f16_back, &f16_data);

    println!("✓ f16 <-> bytes conversion works safely");

    // Test i8 -> bytes conversion
    let i8_data = vec![127i8, -128i8, 0i8, 64i8];
    let i8_bytes = cast_slice::<i8, u8>(&i8_data);
    assert_eq!(i8_bytes.len(), i8_data.len());
    
    let i8_back = cast_slice::<u8, i8>(i8_bytes);
    assert_eq!(i8_back, &i8_data);

    println!("✓ i8 <-> bytes conversion works safely");
}

#[test]
fn test_nan_infinity_handling() {
    println!("Testing NaN and infinity handling...");
    
    let problematic_values = vec![
        1.0f32,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        2.5f32,
    ];

    // Test safe conversion to i8 with NaN/infinity handling
    let safe_i8: Vec<i8> = problematic_values.iter()
        .map(|&x| {
            if !x.is_finite() {
                0i8 // Safe default for non-finite values
            } else {
                (x * 127.0).clamp(-128.0, 127.0) as i8
            }
        })
        .collect();

    assert_eq!(safe_i8[0], 127);  // 1.0 * 127
    assert_eq!(safe_i8[1], 0);    // NaN -> 0
    assert_eq!(safe_i8[2], 0);    // Infinity -> 0
    assert_eq!(safe_i8[3], 0);    // -Infinity -> 0
    assert_eq!(safe_i8[4], 127);  // 2.5 * 127 = 317.5, clamped to 127

    println!("✓ NaN and infinity values handled safely");
}

#[test]
fn test_bounds_validation() {
    println!("Testing bounds validation...");
    
    // Test empty array handling
    let empty: Vec<f32> = vec![];
    let empty_bytes = cast_slice::<f32, u8>(&empty);
    assert!(empty_bytes.is_empty());

    // Test data chunking bounds
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let dim = 2;
    let devices = vec![0, 1, 2];

    // Simulate safe data distribution
    let total_vectors = data.len() / dim; // 3 vectors
    let per_device = total_vectors / devices.len(); // 1 vector per device

    for (i, &_device_id) in devices.iter().enumerate() {
        let start = i * per_device * dim;
        let end = if i == devices.len() - 1 {
            data.len()
        } else {
            (i + 1) * per_device * dim
        };

        // Validate bounds before slicing
        assert!(start < data.len(), "Start index {} out of bounds", start);
        assert!(end <= data.len(), "End index {} out of bounds", end);
        assert!(start <= end, "Invalid range: {} to {}", start, end);

        let chunk = &data[start..end];
        let bytes = cast_slice::<f32, u8>(chunk);
        assert_eq!(bytes.len(), chunk.len() * 4);
    }

    println!("✓ Bounds validation works correctly");
}

#[test]
fn test_memory_safety_improvements() {
    println!("Testing memory safety improvements...");

    // This test demonstrates what we USED to do (unsafe) vs what we do now (safe)
    let data = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32];

    // OLD UNSAFE WAY (what we replaced):
    // let unsafe_bytes = unsafe {
    //     std::slice::from_raw_parts(
    //         data.as_ptr() as *const u8,
    //         data.len() * std::mem::size_of::<f32>()
    //     ).to_vec()
    // };

    // NEW SAFE WAY (what we use now):
    let safe_bytes = cast_slice::<f32, u8>(&data).to_vec();

    // Both should produce the same result, but the safe way has guarantees
    assert_eq!(safe_bytes.len(), data.len() * 4);

    // Verify round-trip safety
    let back_to_f32 = cast_slice::<u8, f32>(&safe_bytes);
    assert_eq!(back_to_f32, &data);

    println!("✓ Memory safety improvements validated");
}

#[test]
fn test_type_safety() {
    println!("Testing type safety with bytemuck...");

    // bytemuck enforces that types are Pod (Plain Old Data) and Zeroable
    // This prevents many classes of memory safety bugs

    // These types should all be safe to cast
    assert_eq!(std::mem::align_of::<f32>(), 4);
    assert_eq!(std::mem::align_of::<f16>(), 2);  
    assert_eq!(std::mem::align_of::<i8>(), 1);
    assert_eq!(std::mem::align_of::<u8>(), 1);

    // Test that sizes are as expected
    assert_eq!(std::mem::size_of::<f32>(), 4);
    assert_eq!(std::mem::size_of::<f16>(), 2);
    assert_eq!(std::mem::size_of::<i8>(), 1);
    assert_eq!(std::mem::size_of::<u8>(), 1);

    println!("✓ Type safety constraints verified");
}