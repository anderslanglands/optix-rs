fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Initializing CUDA...");
    cu::init()?;
    let device_count = cu::Device::get_count()?;
    if device_count == 0 {
        panic!("No CUDA devices found!");
    }

    println!("Initializing OptiX...");
    optix::init()?;

    println!("Successfully initialized OptiX... yay!");

    Ok(())

}