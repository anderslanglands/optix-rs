use crate::{sys, Error, Stream};
type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Copy, Clone)]
pub struct Event {
    pub(crate) inner: sys::CUevent,
}

impl Event {
    pub fn create(flags: EventFlags) -> Result<Event> {
        let mut inner = std::ptr::null_mut();
        unsafe {
            sys::cuEventCreate(&mut inner, flags.bits())
                .to_result()
                .map(|_| Event { inner })
        }
    }

    pub fn record(&self, stream: &Stream) -> Result<()> {
        unsafe { sys::cuEventRecord(self.inner, stream.inner).to_result() }
    }

    pub fn elapsed(&self, since: &Event) -> Result<f32> {
        let mut milliseconds = 0.0f32;
        unsafe {
            sys::cuEventElapsedTime(&mut milliseconds, since.inner, self.inner)
                .to_result()
                .map(|_| milliseconds)
        }
    }

    pub fn synchronize(&self) -> Result<()> {
        unsafe { sys::cuEventSynchronize(self.inner).to_result() }
    }
}

bitflags::bitflags! {
    pub struct EventFlags: u32 {
        const DEFAULT = sys::CUevent_flags_enum_CU_EVENT_DEFAULT;
        const BLOCKING_SYNC = sys::CUevent_flags_enum_CU_EVENT_BLOCKING_SYNC;
        const DISABLE_TIMING = sys::CUevent_flags_enum_CU_EVENT_DISABLE_TIMING;
        const INTERPROCESS = sys::CUevent_flags_enum_CU_EVENT_INTERPROCESS;
    }
}
