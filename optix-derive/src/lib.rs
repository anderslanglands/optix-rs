#![feature(proc_macro_diagnostic)]

extern crate proc_macro;
use proc_macro::TokenStream;
use quote::quote;
use syn::{
    punctuated::Punctuated, token::Comma, Data, DataEnum, DataStruct,
    DeriveInput, Field, Fields, Variant,
};

#[proc_macro_attribute]
pub fn device_shared(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(item as DeriveInput);

    /*
    let fields = match &input.data {
        Data::Struct(DataStruct {
            fields: Fields::Named(fields),
            ..
        }) => &fields.named,
        _ => panic!("expected a struct with named fields"),
    };
    */
    let result = match &input.data {
        Data::Struct(DataStruct {
            fields: Fields::Named(fields),
            ..
        }) => do_struct(&input, &fields.named),
        Data::Enum(DataEnum { variants, .. }) => do_enum(&input, &variants),
        _ => unimplemented!(),
    };

    // panic!("{}", result);

    result.into()
}

fn do_enum(
    input: &DeriveInput,
    variants: &Punctuated<Variant, Comma>,
) -> proc_macro2::TokenStream {
    let name = &input.ident;

    let variant_idents: Vec<_> = variants
        .iter()
        .map(|variant| format!("{}", &variant.ident))
        .collect();

    let s_name = format!("{}", name);

    let result = quote! {
        #[repr(u32)]
        #[allow(dead_code)]
        #[derive(Copy, Clone, PartialEq, PartialOrd)]
        #input

        impl DeviceShareable for #name {
            type Target = u32;

            fn to_device(&self) -> u32 {
                *self as u32
            }

            fn cuda_type() -> String {
                #s_name.into()
            }

            fn cuda_decl() -> String {
                let mut s = format!("enum class {}: u32 {{\n", #s_name);

                #(
                    s = format!("{} {},\n", s, #variant_idents);
                )*

                s = format!("{} }};", s);

                s
            }
        }
    };

    result
}

fn do_struct(
    input: &DeriveInput,
    fields: &Punctuated<Field, Comma>,
) -> proc_macro2::TokenStream {
    let name = &input.ident;
    let d_name = proc_macro2::Ident::new(
        &format!("{}D", input.ident),
        proc_macro2::Span::call_site(),
    );
    let generics = &input.generics;

    let field_vis = fields.iter().map(|field| &field.vis);
    let field_name = fields.iter().map(|field| &field.ident);
    let field_type = fields.iter().map(|field| &field.ty);

    let result = quote! {
        // output original struct first
        #input

        // now device-compatible struct
        #[repr(C)]
        #[derive(Copy, Clone)]
        pub struct #d_name#generics {
            #(
                #field_vis #field_name: <#field_type as DeviceShareable>::Target,
            )*
        }
    };

    let s_field_name = fields
        .iter()
        .map(|field| {
            format!(
                "{}",
                match &field.ident {
                    Some(s) => s,
                    None => panic!("no ident"),
                }
            )
        })
        .collect::<Vec<_>>();

    let s_name = format!("{}", name);
    let field_type = fields.iter().map(|field| &field.ty);

    let cuda_decl = quote! {
        fn cuda_type() -> String {
            #s_name.into()
        }
        fn cuda_decl() -> String {
            let mut s =   format!("struct {} {{", #s_name);
            #(
                s = format!("{} {} {};", s, <#field_type as DeviceShareable>::cuda_type(), #s_field_name);
            )*
            s = format!("{} }};", s);
            s
        }
    };

    let field_name = fields.iter().map(|field| &field.ident);

    let result = quote! {
        #result

        // now impl DeviceShareable for the original struct
        impl#generics DeviceShareable for #name#generics {
            type Target = #d_name#generics;
            fn to_device(&self) -> Self::Target {
                #d_name {
                    #(
                        #field_name: self.#field_name.to_device(),
                    )*
                }
            }
            #cuda_decl
        }
    };

    result
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
