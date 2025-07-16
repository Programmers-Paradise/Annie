#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Precision {
    Fp32,
    Fp16,
    Int8,
}

impl Precision {
    pub fn element_size(&self) -> usize {
        match self {
            Precision::Fp32 => 4,
            Precision::Fp16 => 2,
            Precision::Int8 => 1,
        }
    }

    pub fn kernel_suffix(&self) -> &'static str {
        match self {
            Precision::Fp32 => "fp32",
            Precision::Fp16 => "fp16",
            Precision::Int8 => "int8",
        }
    }
}