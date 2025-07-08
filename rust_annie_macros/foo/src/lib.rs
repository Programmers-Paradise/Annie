use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput, AttributeArgs, NestedMeta, Lit, Meta};
use syn::spanned::Spanned;
use proc_macro2::Span;
use syn::Ident;

#[proc_macro_attribute]
pub fn py_annindex(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as AttributeArgs);
// Removed unused backend_name variable
    let mut distance_metric = quote! { crate::metrics::Distance::Euclidean };

    // Parse attributes
    for arg in args {
        if let NestedMeta::Meta(Meta::NameValue(nv)) = arg {
// Removed parsing logic for unused backend attribute
            } else if nv.path.is_ident("distance") {
                if let Lit::Str(s) = &nv.lit {
                    distance_metric = match s.value().as_str() {
                        "Euclidean" => quote! { crate::metrics::Distance::Euclidean },
                        "Cosine" => quote! { crate::metrics::Distance::Cosine },
                        "Manhattan" => quote! { crate::metrics::Distance::Manhattan },
                        "Chebyshev" => quote! { crate::metrics::Distance::Chebyshev },
                        _ => distance_metric,
                    };
                }
            }
        }
    }

// Removed unused backend variable
    let input = parse_macro_input!(item as DeriveInput);
    let name = &input.ident;
    
    // Create new identifier for Python struct
    let py_name = Ident::new(&format!("Py{}", name), Span::call_site());
    
    let expanded = quote! {
        #input
        
        #[pyo3::pyclass]
        pub struct #py_name {
            inner: #name,
        }

        #[pyo3::pymethods]
        impl #py_name {
            #[new]
            fn new(dim: usize) -> Self {
                #py_name {
                    inner: #name::new(dim, #distance_metric),
                }
            }

            fn add(
                &mut self,
                py: pyo3::Python,
                data: numpy::PyReadonlyArray2<f32>,
                ids: numpy::PyReadonlyArray1<i64>
            ) -> pyo3::PyResult<()> {
                let dims = self.inner.dims();
                let shape = data.shape();
                if shape.len() != 2 || shape[1] != dims {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        format!("Input data must be of shape (n, {})", dims),
                    ));
                }

                let data_slice = data.as_slice()?;
                let ids_slice = ids.as_slice()?;
                let n_vectors = shape[0];

                if ids_slice.len() != n_vectors {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "ids length must match number of vectors",
                    ));
                }

                if let Some(batch_insert) = self.inner.batch_insert {
                    batch_insert(data_slice, ids_slice, dims);
                } else {
                    for (i, vector) in data_slice.chunks_exact(dims).enumerate() {
                        self.inner.insert(vector, ids_slice[i]);
                    }
                }
                Ok(())
            }

            fn build(&mut self) {
                self.inner.build();
            }

            fn search(&self, vector: Vec<f32>, k: usize) -> Vec<i64> {
                self.inner.search(&vector, k)
            }

            fn save(&self, path: String) -> PyResult<()> {
                Self::validate_path(&path)?;
                self.inner.save(&path);
                Ok(())
            }
            
            #[staticmethod]
            fn load(path: String) -> pyo3::PyResult<Self> {
                if path.contains("..") || path.starts_with('/') || path.starts_with("\\") {
                    return Err(pyo3::exceptions::PyValueError::new_err("Invalid file path"));
                }
                match #name::load(&path) {
                    Ok(inner) => Ok(#py_name { inner }),
                    Err(e) => Err(pyo3::exceptions::PyIOError::new_err(e.to_string())),
                }
            }

            fn validate_path(path: &str) -> PyResult<()> {
                if path.contains("..") || path.starts_with('/') || path.starts_with("\\") {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid file path"));
                }
                Ok(())
            }
        }
    };

    TokenStream::from(expanded)
}