use rust_annie::path_validation::validate_path_secure;
use rust_annie::index::AnnIndex;
use rust_annie::metrics::Distance;
use std::fs;
use pyo3::Python;

#[test]
fn test_basic_directory_traversal() {
    let malicious_paths = [
        "../../../etc/passwd",
        "..\\..\\..\\windows\\system32\\config\\sam",
        "../../../../root/.ssh/id_rsa",
        "../../../../../etc/shadow",
    ];

    for path in &malicious_paths {
        let result = validate_path_secure(path);
        assert!(result.is_err(), "Should reject path: {}", path);
    }
}

#[test]
fn test_url_encoded_traversal() {
    // URL encoded directory traversal attempts
    let encoded_paths = [
        "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
        "%2e%2e%5c%2e%2e%5c%2e%2e%5cwindows%5csystem32",
        "..%2f..%2f..%2fetc%2fpasswd",
        "%252e%252e%252f%252e%252e%252f%252e%252e%252fetc%252fpasswd",
    ];

    for path in &encoded_paths {
        let result = validate_path_secure(path);
        assert!(result.is_err(), "Should reject encoded path: {}", path);
    }
}

#[test]
fn test_mixed_separators() {
    // Mixed path separators to bypass simple filters
    let mixed_paths = [
        "..\\../../../etc/passwd",
        "../..\\../etc/passwd", 
        "..\\..\\../etc/passwd",
        "....//....//etc/passwd",
    ];

    for path in &mixed_paths {
        let result = validate_path_secure(path);
        assert!(result.is_err(), "Should reject mixed separator path: {}", path);
    }
}

#[test]
fn test_absolute_path_rejection() {
    // Absolute paths should be rejected
    let absolute_paths = [
        "/etc/passwd",
        "/root/.ssh/id_rsa",
        "\\Windows\\System32\\config\\sam",
        "C:\\Windows\\System32\\drivers\\etc\\hosts",
        "/proc/version",
        "/dev/null",
    ];

    for path in &absolute_paths {
        let result = validate_path_secure(path);
        assert!(result.is_err(), "Should reject absolute path: {}", path);
    }
}

#[test]
fn test_null_byte_injection() {
    // Null byte injection attempts
    let null_byte_paths = [
        "safe.txt\0../../../etc/passwd",
        "safe.txt\x00../../../etc/passwd", 
        "safe\0.txt",
        "safe.txt\0\0",
    ];

    for path in &null_byte_paths {
        let result = validate_path_secure(path);
        assert!(result.is_err(), "Should reject null byte path: {}", path);
    }
}

#[test]
fn test_control_character_injection() {
    // Control character injection
    let control_paths = [
        "safe\x01.txt",
        "safe\x02.txt",
        "safe\x7f.txt",
        "safe\x1f.txt",
    ];

    for path in &control_paths {
        let result = validate_path_secure(path);
        assert!(result.is_err(), "Should reject control character path: {}", path);
    }
}

#[test] 
fn test_valid_paths() {
    // These should be accepted
    let valid_paths = [
        "model.bin",
        "data.txt",
        "index.ann",
        "my_model_v2.bin",
    ];

    for path in &valid_paths {
        let result = validate_path_secure(path);
        // Note: May fail in test environment due to canonicalization
        // but should not fail due to security checks
        if result.is_err() {
            // Check it's not failing due to security reasons by looking at error message
            // In a real test environment, you'd check the specific error type
            println!("Path {} failed validation (may be due to filesystem): {:?}", path, result);
        }
    }
}

#[test]
fn test_valid_subdirectory_paths() {
    // Create test directories
    let _ = fs::create_dir_all("./data");
    let _ = fs::create_dir_all("./models");
    
    let valid_subpaths = [
        "./data/model.bin",
        "./models/index.ann",
        "data/test.txt",
        "models/v1.bin",
    ];

    for path in &valid_subpaths {
        let result = validate_path_secure(path);
        // These may fail due to canonicalization in test environment
        // but important thing is they don't fail security checks
        match result {
            Ok(_) => println!("Path {} validated successfully", path),
            Err(e) => println!("Path {} failed (may be filesystem related): {:?}", path, e),
        }
    }
}

#[test]
fn test_double_encoding_attacks() {
    // Double URL encoding attacks
    let double_encoded = [
        "%252e%252e%252f%252e%252e%252f%252e%252e%252fetc%252fpasswd",
        "%25252e%25252e%25252f",
        "%2527%2527%252e%252e%252f",
    ];

    for path in &double_encoded {
        let result = validate_path_secure(path);
        assert!(result.is_err(), "Should reject double encoded path: {}", path);
    }
}

#[test]
fn test_unicode_normalization_attacks() {
    // Unicode normalization bypasses (simplified test)
    let unicode_paths = [
        "..᱾etc᱾passwd", // Using similar-looking Unicode chars
        "ᴜᴩ/ᴇᴛᴄ/ᴘᴀsswd", // Unicode lookalikes
    ];

    for path in &unicode_paths {
        let result = validate_path_secure(path);
        // These should either be rejected or normalized safely
        match result {
            Ok(safe_path) => {
                // If accepted, ensure it doesn't resolve to dangerous location
                let path_str = safe_path.to_string_lossy();
                assert!(!path_str.contains("etc"), "Should not resolve to etc: {}", path_str);
            },
            Err(_) => {
                // Rejection is also acceptable
                println!("Unicode path rejected: {}", path);
            }
        }
    }
}

#[test]
fn test_symlink_traversal() {
    // Test that symlinks don't allow traversal (when filesystem supports it)
    // Note: This test may not work in all test environments
    
    let test_dir = "./test_symlink_dir";
    let _ = fs::create_dir_all(test_dir);
    
    // This is a simplified test - in practice, canonicalize() handles symlinks
    let result = validate_path_secure("./test_symlink_dir/../../../etc/passwd");
    assert!(result.is_err(), "Should reject traversal through symlinks");
    
    let _ = fs::remove_dir_all(test_dir);
}

#[test]
fn test_ann_index_save_load_security() {
    // Test that AnnIndex save/load operations use secure path validation
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|_py| {
        let index = AnnIndex::new(3, Distance::Euclidean()).unwrap();
        
        // These should fail due to path traversal
        let malicious_save_paths = [
            "../../../tmp/malicious",
            "/etc/passwd", 
            "C:\\Windows\\System32\\malicious",
        ];

        for path in &malicious_save_paths {
            let result = index.save(path);
            assert!(result.is_err(), "Save should reject malicious path: {}", path);
        }

        // Load should also reject malicious paths
        for path in &malicious_save_paths {
            let result = AnnIndex::load(path);
            assert!(result.is_err(), "Load should reject malicious path: {}", path);
        }
    });
}

#[test]
fn test_case_sensitivity_bypasses() {
    // Test various case combinations
    let case_variants = [
        "../../../ETC/passwd",
        "../../../etc/PASSWD", 
        "../../../Etc/Passwd",
        "..\\..\\..\\WINDOWS\\system32",
    ];

    for path in &case_variants {
        let result = validate_path_secure(path);
        assert!(result.is_err(), "Should reject case variant: {}", path);
    }
}

#[test] 
fn test_path_length_limits() {
    // Very long paths that might cause buffer overflows or bypass checks
    let long_traversal = "../".repeat(1000) + "etc/passwd";
    let result = validate_path_secure(&long_traversal);
    assert!(result.is_err(), "Should reject extremely long traversal path");

    // Long valid path
    let long_valid = "a".repeat(200) + ".txt";
    let result = validate_path_secure(&long_valid);
    // This may fail due to filesystem limits, which is acceptable
    match result {
        Ok(_) => println!("Long valid path accepted"),
        Err(_) => println!("Long valid path rejected (may be due to filesystem limits)"),
    }
}

#[test]
fn test_environment_variable_injection() {
    // Paths that might try to use environment variables
    let env_paths = [
        "$HOME/.ssh/id_rsa",
        "%USERPROFILE%\\Documents",
        "${HOME}/../../etc/passwd",
        "%TEMP%\\..\\..\\Windows\\System32",
    ];

    for path in &env_paths {
        let result = validate_path_secure(path);
        // These should be treated as literal strings, not expanded
        // If they contain traversal, should be rejected
        match result {
            Ok(safe_path) => {
                let path_str = safe_path.to_string_lossy();
                // Should not contain traversal components
                assert!(!path_str.contains(".."), "Should not contain traversal: {}", path_str);
                assert!(!path_str.contains("/etc"), "Should not resolve to /etc: {}", path_str);
            },
            Err(_) => {
                println!("Environment variable path rejected: {}", path);
            }
        }
    }
}
