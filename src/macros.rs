use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput, AttributeArgs, NestedMeta, Lit, Meta};

#[proc_macro_attribute]
pub fn py_annindex(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as AttributeArgs);
    let mut backend_name = None;
    let mut distance_metric = quote! { crate::metrics::Distance::Euclidean };

    // Parse attributes
    for arg in args {
        if let NestedMeta::Meta(Meta::NameValue(nv)) = arg {
            if nv.path.is_ident("backend") {
                if let Lit::Str(s) = &nv.lit {
                    backend_name = Some(s.value());
                }
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

    let backend = backend_name.expect("Must specify backend name");
    let input = parse_macro_input!(item as DeriveInput);
    let name = &input.ident;
    
    let expanded = quote! {
        #input
        
        #[pyclass]
        pub struct Py#name {
            inner: #name,
        }

        #[pymethods]
        impl Py#name {
            #[new]
            fn new(dim: usize) -> Self {
                Py#name {
                    inner: #name::new(dim, #distance_metric),
                }
            }

            fn add(
                &mut self,
                py: Python,
                data: PyReadonlyArray2<f32>,
                ids: PyReadonlyArray1<i64>
            ) -> PyResult<()> {
                let dims = self.inner.dims();
                let shape = data.shape();
                if shape.len() != 2 || shape[1] != dims {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        format!("Input data must be of shape (n, {})", dims),
                    ));
                }

                let data_slice = data.as_slice()?;
                let ids_slice = ids.as_slice()?;
                let n_vectors = shape[0];

                if ids_slice.len() != n_vectors {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "ids length must match number of vectors",
                    ));
                }

                for (i, vector) in data_slice.chunks_exact(dims).enumerate() {
                    self.inner.insert(vector, ids_slice[i]);
                }
                Ok(())
            }

            fn build(&mut self) {
                self.inner.build();
            }

            fn search(&self, vector: Vec<f32>, k: usize) -> Vec<usize> {
                self.inner.search(&vector, k)
            }

            fn save(&self, path: String) {
                self.inner.save(&path);
            }

            #[staticmethod]
            fn load(path: String) -> PyResult<Self> {
                match #name::load(&path) {
                    Ok(inner) => Ok(Py#name { inner }),
                    Err(e) => Err(PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string())),
                }
            }
        }
    };

    TokenStream::from(expanded)
}