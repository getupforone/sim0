// #[derive(Debug, Clone, Copy)]
use serde::{Serialize, Deserialize};
#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub struct Config{
    // pub model_name: String,
    pub target_channel: usize,
    pub d_model: usize,
    pub d_latent: usize,
    pub input_window: usize,
    pub output_window: usize,
    pub batch_size: usize,
    pub sampling_rate: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub dropout: f32,
    pub d_ff: usize,
    pub n_epochs: usize,
    pub grad_norm_clip: f64,
    pub input_dim: usize,
    pub output_dim: usize,

    // pub eval_freq: usize,

    pub learning_rate: f64,
    pub gamma: f64,
    pub num_gpus: usize,
    pub rng_seed: usize,
    pub emb_dim: usize,
    pub qkv_bias: bool,
    pub eval_iter: usize,
    pub eval_freq: usize,
    pub chck_freq: usize,

}
impl Config {
    pub fn trfm() -> Self {
        Self {
            // model_name: String::from("trfm"),
            target_channel: 16,
            d_model: 512,
            d_latent: 64,
            input_window: 16,
            output_window: 16,
            
            batch_size: 32,
            sampling_rate: 100,
            n_layers: 3,
            n_heads: 8,
            dropout: 0.1,
            d_ff: 2048,
            n_epochs: 100,
            grad_norm_clip: 0.5,
            input_dim: 50,
            output_dim: 50,

            // eval_freq: 10, 

            learning_rate: 0.0005_f64,
            gamma: 0.95_f64,
            num_gpus: 1,
            rng_seed: 42,
            emb_dim: 512,
            eval_iter: 50,
            eval_freq: 1000,
            chck_freq: 1,
            qkv_bias : false,
        }
    }
    pub fn rectrfm() -> Self {
        Self {
            // model_name: String::from("rectrfm"),
            target_channel: 16,
            d_model: 64,
            d_latent: 8,
            input_window: 16,
            output_window: 16,
            
            batch_size: 32,
            sampling_rate: 100,
            n_layers: 1,
            n_heads: 2,
            dropout: 0.1,
            d_ff: 128,
            n_epochs: 100,
            grad_norm_clip: 0.7_f64,
            input_dim: 50,
            output_dim: 50,

            // eval_freq: 10, 

            learning_rate: 0.0005_f64,
            gamma: 0.95_f64,
            num_gpus: 1,
            rng_seed: 42,
            emb_dim: 512,
            eval_iter: 64,
            eval_freq: 1000,
            chck_freq: 1,
            qkv_bias : false,
        }
    }
    // org one
    // pub fn rectrfm() -> Self {
    //     Self {
    //         // model_name: String::from("rectrfm"),
    //         target_channel: 16,
    //         d_model: 512,
    //         d_latent: 64,
    //         input_window: 16,
    //         output_window: 16,
            
    //         batch_size: 32,
    //         sampling_rate: 100,
    //         n_layers: 2,
    //         n_heads: 8,
    //         dropout: 0.1,
    //         d_ff: 2048,
    //         n_epochs: 100,
    //         grad_norm_clip: 0.7_f64,
    //         input_dim: 50,
    //         output_dim: 50,

    //         // eval_freq: 10, 

    //         learning_rate: 0.0005_f64,
    //         gamma: 0.95_f64,
    //         num_gpus: 1,
    //         rng_seed: 42,
    //         emb_dim: 512,
    //         eval_iter: 64,
    //         eval_freq: 1000,
    //         chck_freq: 1,
    //         qkv_bias : false,
    //     }
    // }
}
