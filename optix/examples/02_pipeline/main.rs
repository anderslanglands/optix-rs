#[macro_use]
extern crate enum_primitive;

use num::FromPrimitive;

mod sample_renderer;
use optix::cuda::{TaggedAllocator, TaggedMallocator};
use sample_renderer::{MemTags, SampleRenderer};

use optix::math::*;

fn main() {
    let alloc = TaggedMallocator::new();
    let mut sample = SampleRenderer::new(v2i32(1200, 1024), &alloc).unwrap();
    sample.render();

    println!("Total allocated: {}", alloc.total_allocated());
    let tags = alloc.tag_allocations();
    for (tag, size) in tags.iter() {
        println!("{:?}: {}", MemTags::from_u64(*tag).unwrap(), size);
    }
}
