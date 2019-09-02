#[macro_use]
extern crate derive_more;

mod sample_renderer;
use sample_renderer::SampleRenderer;

use nalgebra_glm::IVec2;

fn main() {
    let mut sample = SampleRenderer::new(IVec2::new(1200, 1024)).unwrap();
    sample.render();
}
