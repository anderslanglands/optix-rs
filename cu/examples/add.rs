fn main() -> Result<(), Box<dyn std::error::Error>> {
    /*
    cu::init()?;
    let device = cu::Device::get(0)?;
    let _ctx = device.ctx_create(
        cu::ContextFlags::SCHED_AUTO | cu::ContextFlags::MAP_HOST,
    )?;

    let module = cu::Module::load_string(include_str!(concat!(
        env!("OUT_DIR"),
        "/add.ptx"
    )))?;

    let function = module.get_function("sum")?;
    let stream = cu::Stream::create(cu::StreamFlags::NON_BLOCKING)?;

    let buf_x = cu::Buffer::from_slice(&[2.0f32; 10])?;
    let buf_y = cu::Buffer::from_slice(&[4.0f32; 10])?;
    let mut out = vec![0.0f32; 10];
    let buf_out = cu::Buffer::from_slice(&out)?;
    let len = 10u32;

    cu::launch!(
        function,
        1,
        1,
        0,
        stream,
        buf_x.device_ptr(),
        buf_y.device_ptr(),
        buf_out.device_ptr(),
        len
    )?;

    cu::Context::synchronize()?;

    buf_out.copy_to_slice(&mut out)?;

    assert_eq!(out, [6.0f32; 10]);
    println!("{:?}", out);

    */
    Ok(())
}
