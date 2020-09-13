#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Dim {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl Dim {
    pub fn new_1d(x: u32) -> Dim {
        Dim { x, y: 1, z: 1 }
    }

    pub fn new_2d(x: u32, y: u32) -> Dim {
        Dim { x, y, z: 1 }
    }

    pub fn new_3d(x: u32, y: u32, z: u32) -> Dim {
        Dim { x, y, z }
    }
}

impl From<u32> for Dim {
    fn from(x: u32) -> Dim {
        Dim::new_1d(x)
    }
}

impl From<(u32, u32)> for Dim {
    fn from(t: (u32, u32)) -> Dim {
        Dim::new_2d(t.0, t.1)
    }
}

impl From<(u32, u32, u32)> for Dim {
    fn from(t: (u32, u32, u32)) -> Dim {
        Dim::new_3d(t.0, t.1, t.2)
    }
}

#[macro_export]
macro_rules! launch {
    ($function:ident, $grid_dim:expr, $block_dim:expr, $shared_mem_bytes:expr, $stream:expr, $( $arg:expr),* ) => {
        $stream.launch_kernel(
            $function,
            $grid_dim,
            $block_dim,
            $shared_mem_bytes,
            &[
                $(
                    &$arg as *const _ as *mut std::ffi::c_void,
                )*
                std::ptr::null_mut() 
            ]
        )
    };
}
