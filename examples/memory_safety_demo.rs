// Demonstration of memory safety improvements in GPU backend
// This example shows the before/after of our memory safety fixes

use bytemuck::cast_slice;
use half::f16;

fn main() {
    println!("🔒 Memory Safety Improvements Demo");
    println!("==================================");
    
    // Demonstrate safe type conversions
    demo_safe_conversions();
    
    // Demonstrate NaN/infinity handling
    demo_nan_infinity_handling();
    
    // Demonstrate bounds checking
    demo_bounds_checking();
    
    println!("\n✅ All memory safety demonstrations completed successfully!");
}

fn demo_safe_conversions() {
    println!("\n📊 Safe Type Conversions");
    println!("------------------------");
    
    let data = vec![1.0f32, 2.5f32, -3.14f32, 0.0f32];
    
    // OLD UNSAFE WAY (commented out - what we used to do):
    /*
    let unsafe_bytes = unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const u8,
            data.len() * std::mem::size_of::<f32>()
        ).to_vec()
    };
    */
    
    // NEW SAFE WAY (what we do now):
    let safe_bytes = cast_slice::<f32, u8>(&data).to_vec();
    
    println!("Original f32 data: {:?}", data);
    println!("Safely converted to {} bytes", safe_bytes.len());
    
    // Verify round-trip safety
    let back_to_f32 = cast_slice::<u8, f32>(&safe_bytes);
    println!("Round-trip back to f32: {:?}", back_to_f32);
    assert_eq!(back_to_f32, &data);
    println!("✓ Round-trip conversion verified");
    
    // Test f16 conversion
    let f16_data: Vec<f16> = data.iter().map(|&x| f16::from_f32(x)).collect();
    let f16_bytes = cast_slice::<f16, u8>(&f16_data).to_vec();
    println!("f16 conversion: {} bytes", f16_bytes.len());
    
    // Test i8 conversion
    let i8_data: Vec<i8> = data.iter()
        .map(|&x| (x * 127.0).clamp(-128.0, 127.0) as i8)
        .collect();
    let i8_bytes = cast_slice::<i8, u8>(&i8_data).to_vec();
    println!("i8 conversion: {} bytes", i8_bytes.len());
}

fn demo_nan_infinity_handling() {
    println!("\n🚫 NaN/Infinity Handling");
    println!("------------------------");
    
    let problematic_data = vec![
        1.0f32,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        2.5f32,
    ];
    
    println!("Input data with problematic values: {:?}", problematic_data);
    
    // Safe conversion with NaN/infinity handling
    let safe_i8: Vec<i8> = problematic_data.iter()
        .map(|&x| {
            if !x.is_finite() {
                println!("  Detected non-finite value: {} -> converted to 0", x);
                0i8 // Safe default for non-finite values
            } else {
                let scaled = (x * 127.0).clamp(-128.0, 127.0) as i8;
                println!("  Finite value: {} -> {}", x, scaled);
                scaled
            }
        })
        .collect();
    
    println!("Safe i8 conversion result: {:?}", safe_i8);
    println!("✓ All non-finite values handled safely");
}

fn demo_bounds_checking() {
    println!("\n📏 Bounds Checking");
    println!("------------------");
    
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let dim = 2;
    let num_devices = 3;
    
    println!("Data: {:?} (dim={})", data, dim);
    println!("Distributing across {} devices", num_devices);
    
    // Simulate safe data distribution with bounds checking
    let total_vectors = data.len() / dim;
    println!("Total vectors: {}", total_vectors);
    
    if data.len() % dim != 0 {
        println!("⚠️  Warning: Data length {} not divisible by dimension {}", data.len(), dim);
    }
    
    let per_device = total_vectors / num_devices;
    println!("Vectors per device: {}", per_device);
    
    for device_id in 0..num_devices {
        let start = device_id * per_device * dim;
        let end = if device_id == num_devices - 1 {
            data.len()
        } else {
            (device_id + 1) * per_device * dim
        };
        
        // Bounds validation (what we added for safety)
        if start >= data.len() {
            println!("❌ Device {}: Invalid start index {} >= {}", device_id, start, data.len());
            continue;
        }
        if end > data.len() {
            println!("❌ Device {}: Invalid end index {} > {}", device_id, end, data.len());
            continue;
        }
        if start > end {
            println!("❌ Device {}: Invalid range {} > {}", device_id, start, end);
            continue;
        }
        
        let chunk = &data[start..end];
        let bytes = cast_slice::<f32, u8>(chunk);
        
        println!("✓ Device {}: indices {}..{}, {} values, {} bytes", 
                device_id, start, end, chunk.len(), bytes.len());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_safety_invariants() {
        // Test that our safe conversions maintain invariants
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        
        // Safe conversion should never panic
        let bytes = cast_slice::<f32, u8>(&data);
        assert_eq!(bytes.len(), data.len() * 4);
        
        // Round-trip should be lossless for f32
        let back = cast_slice::<u8, f32>(bytes);
        assert_eq!(back, &data);
    }
    
    #[test]
    fn test_nan_safety() {
        let data = vec![f32::NAN, f32::INFINITY, 1.0];
        
        // This should never panic or produce undefined behavior
        let safe_i8: Vec<i8> = data.iter()
            .map(|&x| if x.is_finite() { (x * 127.0) as i8 } else { 0 })
            .collect();
            
        assert_eq!(safe_i8[0], 0);  // NaN -> 0
        assert_eq!(safe_i8[1], 0);  // Infinity -> 0
        assert_eq!(safe_i8[2], 127); // 1.0 -> 127
    }
}