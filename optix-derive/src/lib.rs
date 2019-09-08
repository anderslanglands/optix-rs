#![feature(proc_macro_diagnostic)]

extern crate proc_macro;
use proc_macro::{Diagnostic, TokenStream};
use quote::{quote, ToTokens};
use syn::{parse2, Data, DataStruct, DeriveInput, Fields};

#[proc_macro_attribute]
pub fn device_shared(attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(item as DeriveInput);

    let attrs = &input.attrs.iter();
    let vis = &input.vis;
    let name = &input.ident;
    let d_name = proc_macro2::Ident::new(
        &format!("{}D", input.ident),
        proc_macro2::Span::call_site(),
    );
    let data = &input.data;

    let fields = match &input.data {
        Data::Struct(DataStruct {
            fields: Fields::Named(fields),
            ..
        }) => &fields.named,
        _ => panic!("expected a struct with named fields"),
    };

    let field_vis = fields.iter().map(|field| &field.vis);
    let field_name = fields.iter().map(|field| &field.ident);
    let field_type = fields.iter().map(|field| &field.ty);

    let result = quote! {
        // output original struct first
        #input

        // now device-compatible struct
        #[repr(C)]
        #[derive(Copy, Clone)]
        pub struct #d_name {
            #(
                #field_vis #field_name: <#field_type as DeviceShareable>::Target,
            )*
        }
    };

    let field_name = fields.iter().map(|field| &field.ident);

    let result = quote! {
        #result

        // now impl DeviceShareable for the original struct
        impl DeviceShareable for #name {
            type Target = #d_name;
            fn to_device(&self) -> Self::Target {
                #d_name {
                    #(
                        #field_name: self.#field_name.to_device(),
                    )*
                }
            }
        }
    };

    // panic!("{}", result);

    result.into()
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
