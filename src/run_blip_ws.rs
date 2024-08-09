use std::io::Cursor;
use axum::body::Bytes;
use candle_core::{Device, DType, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::blip;
use tokenizers::Tokenizer;
use crate::token_output_stream::TokenOutputStream;
use anyhow::Error as E;

enum Model {
    M(blip::BlipForConditionalGeneration),
}

impl Model {
    fn text_decoder_forward(&mut self, xs: &Tensor, img_xs: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Self::M(m) => m.text_decoder().forward(xs, img_xs),
        }
    }
}

const SEP_TOKEN_ID: u32 = 102;

pub fn run_blip_ws(image: Vec<u8>, ) -> anyhow::Result<String> {
    let model_file = {
        let api = hf_hub::api::sync::Api::new()?;
        let api = api.repo(hf_hub::Repo::with_revision(
            "Salesforce/blip-image-captioning-large".to_string(),
            hf_hub::RepoType::Model,
            "refs/pr/18".to_string(),
        ));
        api.get("model.safetensors")?
    };
    let tokenizer = {
        let api = hf_hub::api::sync::Api::new()?;
        let api = api.model("Salesforce/blip-image-captioning-large".to_string());
        api.get("tokenizer.json")?
    };
    let tokenizer = Tokenizer::from_file(tokenizer).map_err(E::msg)?;
    let mut tokenizer = TokenOutputStream::new(tokenizer);
    let mut logits_processor =
        candle_transformers::generation::LogitsProcessor::new(1337, None, None);

    let config = blip::Config::image_captioning_large();

    let device = Device::Cpu;
    let (image_embeds, device, mut model) = {
        let image = load_image_ws(image)?.to_device(&device)?;
        println!("loaded image {image:?}");

        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], DType::F32, &device)? };
        let model = blip::BlipForConditionalGeneration::new(&config, vb)?;
        let image_embeds = image.unsqueeze(0)?.apply(model.vision_model())?;
        (image_embeds, device, Model::M(model))
    };

    let mut token_ids = vec![30522u32];
    let mut result = String::from("");
    for index in 0..1000 {
        let context_size = if index > 0 { 1 } else { token_ids.len() };
        let start_pos = token_ids.len().saturating_sub(context_size);
        let input_ids = Tensor::new(&token_ids[start_pos..], &device)?.unsqueeze(0)?;
        let logits = model.text_decoder_forward(&input_ids, &image_embeds)?;
        let logits = logits.squeeze(0)?;
        let logits = logits.get(logits.dim(0)? - 1)?;
        let token = logits_processor.sample(&logits)?;
        if token == SEP_TOKEN_ID {
            break;
        }
        token_ids.push(token);
        if let Some(t) = tokenizer.next_token(token)? {
            use std::io::Write;
            print!("{t}");
            result += &*t;
            std::io::stdout().flush()?;
        }
    }
    if let Some(rest) = tokenizer.decode_rest().map_err(E::msg)? {
        print!("{rest}");
        result += &*rest;
    }
    println!();
    Ok(result)
}

/// Loads an image from disk using the image crate, this returns a tensor with shape
/// (3, 384, 384). OpenAI normalization is applied.
pub fn load_image_ws(p: Vec<u8>) -> candle_core::Result<Tensor> {
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