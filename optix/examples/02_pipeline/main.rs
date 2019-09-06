#[macro_use]
extern crate derive_more;

mod sample_renderer;
use sample_renderer::SampleRenderer;

use imath::*;

fn main() {
    let mut sample = SampleRenderer::new(v2i32(1200, 1024)).unwrap();
    sample.render();
}
