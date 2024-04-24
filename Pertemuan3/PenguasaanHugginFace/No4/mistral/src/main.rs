// Mengimpor pustaka-pustaka eksternal dan modul yang diperlukan
use anyhow::{Error as E, Result};
use clap::Parser;

use candle_transformers::models::mistral::{Config, Model as Mistral};
use candle_transformers::models::quantized_mistral::Model as QMistral;

use candle_core::{DType, Device, Tensor};
use candle_examples::token_output_stream::TokenOutputStream;
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

// Definisi enum Model yang memiliki dua varian, yaitu Mistral dan Quantized
enum Model {
    Mistral(Mistral),
    Quantized(QMistral),
}

// Definisi struct TextGeneration yang memiliki beberapa field yang diperlukan untuk melakukan generasi teks dengan model yang telah diload sebelumnya
struct TextGeneration {
    model: Model,
    device: Device,
    tokenizer: TokenOutputStream,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

// Implementasi dari struct TextGeneration yang memiliki beberapa method yang diperlukan untuk melakukan generasi teks dengan model yang telah diload sebelumnya
impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    // Method new yang digunakan untuk membuat instance dari struct TextGeneration dengan parameter yang diperlukan untuk melakukan generasi teks
    fn new(
        model: Model,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        device: &Device,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, temp, top_p);
        Self {
            model,
            tokenizer: TokenOutputStream::new(tokenizer),
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            device: device.clone(),
        }
    }

    // Method run yang digunakan untuk menjalankan generasi teks dengan model yang telah diload sebelumnya dengan prompt yang diberikan dan panjang sample yang diinginkan
    fn run(&mut self, prompt: &str, sample_len: usize) -> Result<()> {
        // Mengimport pustaka-pustaka yang diperlukan untuk melakukan operasi input-output pada terminal
        use std::io::Write;
        self.tokenizer.clear();
        let mut tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        for &t in tokens.iter() {
            if let Some(t) = self.tokenizer.next_token(t)? {
                print!("{t}")
            }
        }
        std::io::stdout().flush()?;

        // Melakukan iterasi sebanyak sample_len untuk menghasilkan teks dengan panjang sample_len yang diinginkan
        let mut generated_tokens = 0usize;
        let eos_token = match self.tokenizer.get_token("</s>") {
            Some(token) => token,
            None => anyhow::bail!("cannot find the </s> token"),
        };
        // Menghitung waktu yang diperlukan untuk melakukan generasi teks dengan model yang telah diload sebelumnya
        let start_gen = std::time::Instant::now();
        for index in 0..sample_len {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = match &mut self.model {
                Model::Mistral(m) => m.forward(&input, start_pos)?,
                Model::Quantized(m) => m.forward(&input, start_pos)?,
            };
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &tokens[start_at..],
                )?
            };

            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            generated_tokens += 1;
            if next_token == eos_token {
                break;
            }
            if let Some(t) = self.tokenizer.next_token(next_token)? {
                print!("{t}");
                std::io::stdout().flush()?;
            }
        }
        // Menghitung waktu yang diperlukan untuk melakukan generasi teks dengan model yang telah diload sebelumnya
        let dt = start_gen.elapsed();
        if let Some(rest) = self.tokenizer.decode_rest().map_err(E::msg)? {
            print!("{rest}");
        }
        std::io::stdout().flush()?;
        println!(
            "\n{generated_tokens} tokens generated ({:.2} token/s)",
            generated_tokens as f64 / dt.as_secs_f64(),
        );
        Ok(())
    }
}

// Definisi struct Args yang digunakan untuk mengatur argumen yang diperlukan untuk menjalankan program ini
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
// Implementasi dari struct Args yang memiliki beberapa field yang diperlukan untuk menjalankan program ini
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    #[arg(long)]
    use_flash_attn: bool,

    #[arg(long)]
    prompt: String,

    /// The temperature used to generate samples.
    #[arg(long)]
    temperature: Option<f64>,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// The length of the sample to generate (in tokens).
    #[arg(long, short = 'n', default_value_t = 10000)]
    sample_len: usize,

    #[arg(long)]
    model_id: Option<String>,

    #[arg(long, default_value = "main")]
    revision: String,

    #[arg(long)]
    tokenizer_file: Option<String>,

    #[arg(long)]
    weight_files: Option<String>,

    #[arg(long)]
    quantized: bool,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,
}

// Implementasi dari fungsi main yang digunakan untuk menjalankan program ini
fn main() -> Result<()> {
    // Mengimport pustaka-pustaka yang diperlukan untuk melakukan tracing pada program ini
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    // Mengimport pustaka-pustaka yang diperlukan untuk melakukan operasi input-output pada terminal
    let args = Args::parse();
    let _guard = if args.tracing {
        // Menggunakan tracing untuk melakukan tracing pada program ini
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };
    // Menampilkan informasi tentang fitur-fitur yang didukung oleh perangkat yang digunakan
    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle_core::utils::with_avx(),
        candle_core::utils::with_neon(),
        candle_core::utils::with_simd128(),
        candle_core::utils::with_f16c()
    );
    // Menampilkan informasi tentang argumen-argumen yang diberikan oleh pengguna program ini
    println!(
        "temp: {:.2} repeat-penalty: {:.2} repeat-last-n: {}",
        args.temperature.unwrap_or(0.),
        args.repeat_penalty,
        args.repeat_last_n
    );

    // Menghitung waktu yang diperlukan untuk melakukan load model yang diberikan oleh pengguna program ini
    let start = std::time::Instant::now();
    let api = Api::new()?;
    let model_id = match args.model_id {
        Some(model_id) => model_id,
        None => {
            if args.quantized {
                "lmz/candle-mistral".to_string()
            } else {
                "mistralai/Mistral-7B-v0.1".to_string()
            }
        }
    };
    // Mengambil model yang diberikan oleh pengguna program ini
    let repo = api.repo(Repo::with_revision(
        model_id,
        RepoType::Model,
        args.revision,
    ));
    // Mengambil tokenizer yang diberikan oleh pengguna program ini
    let tokenizer_filename = match args.tokenizer_file {
        Some(file) => std::path::PathBuf::from(file),
        None => repo.get("tokenizer.json")?,
    };
    // Mengambil weight files yang diberikan oleh pengguna program ini
    let filenames = match args.weight_files {
        Some(files) => files
            .split(',')
            .map(std::path::PathBuf::from)
            .collect::<Vec<_>>(),
        None => {
            if args.quantized {
                vec![repo.get("model-q4k.gguf")?]
            } else {
                candle_examples::hub_load_safetensors(&repo, "model.safetensors.index.json")?
            }
        }
    };
    // Menghitung waktu yang diperlukan untuk melakukan load model yang diberikan oleh pengguna program ini
    println!("retrieved the files in {:?}", start.elapsed());
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    // Menghitung waktu yang diperlukan untuk melakukan load model yang diberikan oleh pengguna program ini
    let start = std::time::Instant::now();
    let config = Config::config_7b_v0_1(args.use_flash_attn);
    let device = candle_examples::device(args.cpu)?;
    let (model, device) = if args.quantized {
        // Mengambil model yang diberikan oleh pengguna program ini
        let filename = &filenames[0];
        let vb =
            candle_transformers::quantized_var_builder::VarBuilder::from_gguf(filename, &device)?;
        let model = QMistral::new(&config, vb)?;
        (Model::Quantized(model), device)
    } else {
        let dtype = if device.is_cuda() {
            DType::BF16
        } else {
            DType::F32
        };
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
        let model = Mistral::new(&config, vb)?;
        (Model::Mistral(model), device)
    };

    // Menghitung waktu yang diperlukan untuk melakukan load model yang diberikan oleh pengguna program ini
    println!("loaded the model in {:?}", start.elapsed());

    // Menghitung waktu yang diperlukan untuk melakukan generasi teks dengan model yang telah diload sebelumnya
    let mut pipeline = TextGeneration::new(
        model,
        tokenizer,
        args.seed,
        args.temperature,
        args.top_p,
        args.repeat_penalty,
        args.repeat_last_n,
        &device,
    );
    // Menjalankan generasi teks dengan model yang telah diload sebelumnya dengan prompt yang diberikan dan panjang sample yang diinginkan
    pipeline.run(&args.prompt, args.sample_len)?;
    // Mengembalikan nilai Ok(()) jika program ini berhasil dijalankan
    Ok(())
}
