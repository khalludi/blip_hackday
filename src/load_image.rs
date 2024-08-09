use candle_core::{Device, DType, Tensor};
use tokio_util::io::StreamReader;
use std::io::Cursor;

/// Loads an image from disk using the image crate, this returns a tensor with shape
/// (3, 384, 384). OpenAI normalization is applied.
pub fn load_image(p: axum::body::Bytes) -> candle_core::Result<Tensor> {
    let img = image::ImageReader::new(Cursor::new(p))
        .with_guessed_format()?
        .decode()
        .map_err(candle_core::Error::wrap)?
        .resize_to_fill(384, 384, image::imageops::FilterType::Triangle);
    let img = img.to_rgb8();
    let data = img.into_raw();
    let data = Tensor::from_vec(data, (384, 384, 3), &Device::Cpu)?.permute((2, 0, 1))?;
    let mean =
        Tensor::new(&[0.48145466f32, 0.4578275, 0.40821073], &Device::Cpu)?.reshape((3, 1, 1))?;
    let std = Tensor::new(&[0.26862954f32, 0.261_302_6, 0.275_777_1], &Device::Cpu)?
        .reshape((3, 1, 1))?;
    (data.to_dtype(DType::F32)? / 255.)?
        .broadcast_sub(&mean)?
        .broadcast_div(&std)
}