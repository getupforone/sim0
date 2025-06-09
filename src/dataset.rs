
use std::{collections::HashMap, hash::Hash};

use hdf5::{File as hFile, Group, Result as hResult};
use serde::{Deserialize, Serialize};
use serde_json;

use candle_core::{DType, Var};
use candle_nn::{linear_b, Optimizer, VarBuilder, VarMap};
use candle_nn::{AdamW, ParamsAdamW};

use polars::prelude::*;
// use ndarray::linspace;
use candle_core::{Device, Result as cResult, Tensor, D,safetensors};
use candle_datasets::{batcher::IterResult2, Batcher};

use anyhow::{anyhow, Result};

use arrow::array::{AsArray, Float32Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use parquet::arrow::arrow_reader::{ParquetRecordBatchReader, ParquetRecordBatchReaderBuilder};
use parquet::arrow::{arrow_reader, ArrowWriter};
use parquet::file::properties::WriterProperties;
use rand::{seq::SliceRandom, thread_rng};
use std::fs::File;
use std::path::Path;
use std::rc::Rc;
use std::sync::Arc;

use tqdm::tqdm;

use rayon::prelude::*;

use crate::savedata::{binData,save_bin, load_bin, get_bin_dir,get_bin_file_path,get_binfile_paths_in_folder};

const MODES: [&str; 3] = ["train", "val", "test"];
// const TAG_NAMES: [&str; 50] = [
//     "PF1L_FLOW_IN",
//     "PF1L_PRES_IN",
//     "PF1L_TEMP_IN",
//     "PF1L_TEMP_OUT1",
//     "PF1L_TEMP_OUT2",
//     "PF1L_TEMP_OUT3",
//     "PF1L_TEMP_OUT4",
//     "PF1L_TEMP_OUT5",
//     "PF1L_TEMP_OUT6",
//     "PF1U_CURRENT",
//     "PF1U_FLOW_IN",
//     "PF1U_PRES_IN",
//     "PF1U_TEMP_IN",
//     "PF1U_TEMP_OUT1",
//     "PF1U_TEMP_OUT2",
//     "PF1U_TEMP_OUT3",
//     "PF1U_TEMP_OUT4",
//     "PF1U_TEMP_OUT5",
//     "PF1U_TEMP_OUT6",
//     "PF2L_FLOW_IN",
//     "PF2L_PRES_IN",
//     "PF2U_CURRENT",
//     "PF2U_FLOW_IN",
//     "PF2U_PRES_IN",
//     "PF3L_CURRENT",
//     "PF3L_FLOW_IN",
//     "PF3L_PRES_IN",
//     "PF3U_CURRENT",
//     "PF3U_FLOW_IN",
//     "PF3U_PRES_IN",
//     "PF4L_CURRENT",
//     "PF4L_FLOW_IN",
//     "PF4L_PRES_IN",
//     "PF4U_CURRENT",
//     "PF4U_FLOW_IN",
//     "PF4U_PRES_IN",
//     "PF5L_CURRENT",
//     "PF5L_FLOW_IN",
//     "PF5L_PRES_IN",
//     "PF5U_CURRENT",
//     "PF5U_FLOW_IN",
//     "PF5U_PRES_IN",
//     "PF6L_CURRENT",
//     "PF6U_CURRENT",
//     "PF6_FLOW_IN",
//     "PF6_PRES_IN",
//     "PF7U_CURRENT",
//     "PF7_FLOW_IN",
//     "PF7_PRES_IN",
//     "PF_PRES_OUT",
// ];
const TAG_NAMES: [&str; 50] = [
    "PF1U_CURRENT",
    "PF2U_CURRENT",
    "PF3U_CURRENT",
    "PF3L_CURRENT",
    "PF4U_CURRENT",
    "PF4L_CURRENT",
    "PF5U_CURRENT",
    "PF5L_CURRENT",
    "PF6U_CURRENT",
    "PF6L_CURRENT",
    "PF7U_CURRENT",
    "PF1L_TEMP_IN",
    "PF1L_TEMP_OUT1",
    "PF1L_TEMP_OUT2",
    "PF1L_TEMP_OUT3",
    "PF1L_TEMP_OUT4",
    "PF1L_TEMP_OUT5",
    "PF1L_TEMP_OUT6",
    "PF1U_TEMP_IN",
    "PF1U_TEMP_OUT1",
    "PF1U_TEMP_OUT2",
    "PF1U_TEMP_OUT3",
    "PF1U_TEMP_OUT4",
    "PF1U_TEMP_OUT5",
    "PF1U_TEMP_OUT6",
    "PF1U_PRES_IN",
    "PF1L_PRES_IN",
    "PF2U_PRES_IN",
    "PF2L_PRES_IN",
    "PF3U_PRES_IN",
    "PF3L_PRES_IN",
    "PF4U_PRES_IN",
    "PF4L_PRES_IN",
    "PF5U_PRES_IN",
    "PF5L_FLOW_IN",
    "PF6_PRES_IN",
    "PF7_PRES_IN",
    "PF_PRES_OUT",
    "PF1U_FLOW_IN",
    "PF1L_FLOW_IN",
    "PF2U_FLOW_IN",
    "PF2L_FLOW_IN",
    "PF3U_FLOW_IN",
    "PF3L_FLOW_IN",
    "PF4U_FLOW_IN",
    "PF4L_FLOW_IN",
    "PF5U_FLOW_IN",
    "PF5L_PRES_IN",
    "PF6_FLOW_IN",
    "PF7_FLOW_IN",
];
struct MinMaxScaler<'a> {
    feature_range: (f32, f32),
    min: Option<Tensor>,
    max: Option<Tensor>,
    data_range: Option<Tensor>,
    scale: Option<Tensor>,
    device: &'a Device,
}

impl<'a> MinMaxScaler<'a> {
    fn new(feature_range: (f32, f32), device: &'a Device) -> Self {
        MinMaxScaler {
            feature_range,
            min: None,
            max: None,
            data_range: None,
            scale: None,
            device,
        }
    }
    // fn partial_fit(&mut self, x: &Tensor) -> Result<()> {
    //     // max와 min 텐서 안전하게 가져오기
    //     let max_tensor = self.max.as_ref()
    //         .ok_or_else(|| candle_core::Error::Msg("max tensor is None".to_string()))?;
    //     let min_tensor = self.min.as_ref()
    //         .ok_or_else(|| candle_core::Error::Msg("min tensor is None".to_string()))?;

    //     // 데이터 범위 계산
    //     let data_range = (max_tensor - min_tensor)?;
    //     self.data_range = Some(data_range.clone());

    //     // 스케일 계산
    //     let scale_factor = (self.feature_range.1 - self.feature_range.0) as f64;
    //     let scale = (Tensor::new(&[scale_factor], &self.device)? / &data_range)?;
    //     self.scale = Some(scale);

    //     Ok(())
    // }
    // pub fn get_device(&self) -> &Device {
    //     &self.device
    // }
    fn partial_fit(&mut self, x: &Tensor) -> Result<()> {
        let current_min = x.min_keepdim(1)?;
        let current_max: Tensor = x.max_keepdim(1)?;
        // println!("[partial_fit]x = {:?}", x);
        // println!("[partial_fit]current_min = {}", current_min);
        // println!("[partial_fit]current_max = {}", current_max);

        if self.min.is_none() {
            self.min = Some(current_min);
            self.max = Some(current_max);
        } else {
            let min_tensor = self
                .min
                .as_ref()
                .ok_or_else(|| candle_core::Error::Msg("max tensor is None".to_string()))?;
            let max_tensor = self
                .max
                .as_ref()
                .ok_or_else(|| candle_core::Error::Msg("min tensor is None".to_string()))?;

            self.min = Some(min_tensor.minimum(&current_min)?);
            self.max = Some(max_tensor.maximum(&current_max)?);
        }
        let min_tensor = self
            .min
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("max tensor is None".to_string()))?;
        let max_tensor = self
            .max
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("min tensor is None".to_string()))?;
        // println!("min_tensor = {:?}/{:?}", min_tensor.get(0)?.get(0),min_tensor.get(0)?.get(1));
        let data_range = (max_tensor - min_tensor)?;
        // println!("[partial_fit]data_range = {:?}", data_range);
        self.data_range = Some(data_range.clone());
        let scale_factor = (self.feature_range.1 - self.feature_range.0) as f32;
        let scale_tensor = Tensor::new(&[scale_factor], self.device)?
            .reshape((1, 1))?
            .broadcast_as(data_range.shape())?;
        let scale = (scale_tensor / &data_range)?;
        self.scale = Some(scale);

        Ok(())
    }

    fn transform(&self, x: &Tensor) -> Result<Tensor> {
        let min_tensor = self
            .min
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("min tensor is None".to_string()))?
            .broadcast_as(x.shape())?;
        let scale_tensor = self
            .scale
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("scale_tensor tensor is None".to_string()))?
            .broadcast_as(x.shape())?;
        let diff = (x - min_tensor)?;
        // .ok_or_else(|| candle_core::Error::Msg("x - min is None".to_string()))?;
        let scaled = (diff * scale_tensor)?;
        let offset = Tensor::new(&[self.feature_range.0], &self.device)?
            .reshape((1, 1))?
            .broadcast_as(scaled.shape());

        let result = (scaled + offset)?;
        Ok(result)
    }

    fn fit_transform(&mut self, x: &Tensor) -> Result<Tensor> {
        self.partial_fit(x)?;
        self.transform(x)
    }
}

fn read_parquet_to_tensors(
    file_path: &str,
    device: &Device,
) -> anyhow::Result<(HashMap<String, Tensor>, HashMap<String, usize>)> {

    println!("[read_parquet_to_tensors] file_path = {}", file_path);
    let file = File::open(file_path)?;

    let reader_builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let mut reader = reader_builder.build()?;


    let mut tensor_map: HashMap<String, Vec<Vec<f32>>> = HashMap::new();
    let mut len_map: HashMap<String, usize> = HashMap::new();
    // let mut tensor_map: HashMap<String, Tensor> = HashMap::new();


    while let Some(batch) = reader.next() {
        let batch = batch?;
        // println!(
        //     "batch num_col = {}, num_row{}",
        //     batch.num_columns(),
        //     batch.num_rows()
        // );
        let t1 = std::time::Instant::now();
        let shotnum_id_col = batch
            .column_by_name("shotnum_id")
            .ok_or_else(|| anyhow::anyhow!("no column shotnum_id"))?
            .as_string::<i32>();
        let len_id_col = batch
            .column_by_name("len_id")
            .ok_or_else(|| anyhow::anyhow!("no column len_id"))?
            .as_string::<i32>();
        let dt1 = t1.elapsed();
        println!( "t1 ({}s)",dt1.as_secs_f64(),);

        let t2 = std::time::Instant::now();
        let schema = batch.schema();
        let feature_columns: Vec<&Float32Array> = (0..batch.num_columns())
            .filter_map(|i| {
                let col_name = schema.field(i).name();
                if col_name.starts_with("feature_") {
                    batch.column(i).as_any().downcast_ref::<Float32Array>()
                } else {
                    None
                }
            })
            .collect();
        let dt2 = t2.elapsed();
        println!( "t2 ({}s)",dt2.as_secs_f64(),);
        // let num_rows = feature_columns.len();
        // let num_cols = feature_columns[0].len();
        // let mut tensor_cols_data = Vec::with_capacity(num_rows*num_cols);
        // for col in feature_columns{
        //     tensor_cols_data.extend_from_slice(col.values());
        // }
        // let tensors_map = Tensor::from_vec(tensor_cols_data, (num_rows,num_cols),&Device::Cpu);
        // 각 레코드 처리
        let t3 = std::time::Instant::now();
        for row_idx in 0..batch.num_rows() {
            // 텐서 ID 추출
            let shotnum_id = shotnum_id_col.value(row_idx).to_string();
            let len_id = len_id_col.value(row_idx).parse::<usize>()?;


            // let row_data: Vec<f32> = feature_columns
            //     .iter()
            //     .map(|col| col.value(row_idx))
            //     .collect();
            let row_data = {
                let mut data = Vec::with_capacity(feature_columns.len());
                data.extend(feature_columns.iter().map(|col| col.value(row_idx)));
                data
            };
          
            // tensor_map
            //     .entry(shotnum_id.clone())
            //     .or_insert_with(Vec::new)
            //     .push(row_data);
            if let Some(vec) = tensor_map.get_mut(&shotnum_id) {
                vec.push(row_data);
            } else {
                tensor_map.insert(shotnum_id.clone(), vec![row_data]);
            }

            len_map.entry(shotnum_id).or_insert(len_id);
        }
        let dt3 = t3.elapsed();
        println!( "t3 ({}s)",dt3.as_secs_f64(),);
    }
    println!("[read_parquet_to_tensors] make tensor_map, len_map done ");

    // Ok(tensor_map)
    // 텐서 맵을 Vec<Tensor>로 변환
    // let tensors: Result<Vec<Tensor>> = tensor_map
    // let tensor_map2: Result<HashMap<String,Tensor>> = tensor_map
    //     .into_iter()
    //     .map(|(key,tensor_data)| {
    //         Ok((key,
    //         Tensor::new(
    //             tensor_data.iter()
    //             .map(|row| row.as_slice())
    //             .collect::<Vec<_>>()
    //             // .as_slice()
    //             ,
    //             &Device::Cpu).unwrap()
    //     ))
    //     })
    //     // .collect::<Result<Vec<_>, _>>()
    //     .collect::<anyhow::Result<HashMap<String,Tensor>>>();
    //     // .map_err(|e| anyhow::Error::new(e));
    let t4 = std::time::Instant::now();
    // let tensor_map2: HashMap<String, Tensor> = tensor_map
    //     .into_iter()
    //     .map(|(key, tensor_data)| {
    //         (
    //             key,
    //             Tensor::new(
    //                 tensor_data
    //                     .iter()
    //                     .map(|row| row.as_slice())
    //                     .collect::<Vec<_>>(), // .as_slice()
    //                 device,
    //             )
    //             .unwrap(),
    //         )
    //     })
    //     // .collect::<Result<Vec<_>, _>>()
    //     .collect::<HashMap<String, Tensor>>();
    let tensor_map2: HashMap<String, Tensor> = tensor_map
        .into_par_iter()
        .map(|(key, tensor_data)| {
            
            let mut data_vec = Vec::with_capacity (tensor_data.len());
            for row in tensor_data.iter(){
                data_vec.push(row.as_slice());
            }
            let tensor = Tensor::new(data_vec, device)
                                            .expect("failed to create tensor");
                // Tensor::new(
                //     tensor_data
                //         .iter()
                //         .map(|row| row.as_slice())
                //         .collect::<Vec<_>>(), // .as_slice()
                //     device,
                // )
                // .unwrap(),
            (key, tensor)
        })
        // .collect::<Result<Vec<_>, _>>()
        .collect::<HashMap<String, Tensor>>();
    let dt4 = t4.elapsed();
    println!( "t4 ({}s)",dt4.as_secs_f64(),);
    println!("[read_parquet_to_tensors] tensor_map2 len = {} done", tensor_map2.len());
    // drop(tensor_map);
    // println!("[read_parquet_to_tensors] drop tensor_map done");
    // for (shotnum, len) in &len_map {
    //     println!("(shotnum/length): ({}/{})", shotnum, len);
    // }

    Ok((tensor_map2, len_map))
}
fn read_bin_to_tensors(
    file_path: &str,
    device: &Device,
) -> anyhow::Result<(HashMap<String, Tensor>, HashMap<String, usize>)> {

    println!("[read_parquet_to_tensors] file_path = {}", file_path);


    let (tensor_map , len_map) = load_bin(file_path)?;
    // let mut len_map: HashMap<String, usize> = HashMap::new();

    println!("[read_parquet_to_tensors] make tensor_map, len_map done ");

    let t4 = std::time::Instant::now();

    let tensor_map2: HashMap<String, Tensor> = tensor_map
        .into_par_iter()
        .map(|(key, tensor_data)| {
            
            let mut data_vec = Vec::with_capacity (tensor_data.len());
            for row in tensor_data.iter(){
                data_vec.push(row.as_slice());
            }
            let tensor = Tensor::new(data_vec, device)
                                            .expect("failed to create tensor");
            (key, tensor)
        })
        .collect::<HashMap<String, Tensor>>();
    let dt4 = t4.elapsed();
    println!( "t4 ({}s)",dt4.as_secs_f64(),);
    println!("[read_parquet_to_tensors] tensor_map2 len = {} done", tensor_map2.len());

    Ok((tensor_map2, len_map))
}
fn truncate_tensor(tensor: &Tensor, min_cols: usize) -> Result<Tensor> {
    let shape = tensor.dims().to_vec();

    let current_cols = shape[1];

    if current_cols <= min_cols {
        return Ok(tensor.clone());
    }

    let tensor_cpu = tensor.to_device(&Device::Cpu)?;
    let tensor_data = tensor_cpu.to_vec2::<f32>()?;

    // 최소 크기로 자르기
    let truncated_data: Vec<Vec<f32>> = tensor_data
        .iter()
        .map(|row| row.iter().take(min_cols).cloned().collect())
        .collect();
    // let truncated_tensor = Tensor::new(
    //     truncated_data.iter().map(|row| row.as_slice()).collect::<Vec<_>>(),
    //     &Device::Cpu
    // )?;
    // 자른 데이터로 새 텐서 생성
    let truncated_tensor = Tensor::new(truncated_data, &Device::Cpu)?;

    Ok(truncated_tensor)
}

fn pad_tensor(tensor: &Tensor, max_cols: usize) -> Result<Tensor> {
    let shape = tensor.dims().to_vec();

    let current_cols = shape[1];

    if current_cols >= max_cols {
        return Ok(tensor.clone());
    }

    let tensor_cpu = tensor.to_device(&Device::Cpu)?;
    let tensor_data = tensor_cpu.to_vec2::<f32>()?;

    let padded_data: Vec<Vec<f32>> = tensor_data
        .iter()
        .map(|row| {
            let mut padded_row = row.clone();
            let last_val = *padded_row.last().unwrap_or(&0.0);
            padded_row.extend(vec![last_val; max_cols - current_cols]);
            padded_row
        })
        .collect();

    let padded_tensor = Tensor::new(padded_data, &Device::Cpu)?;

    Ok(padded_tensor)
}
fn div_string_vec_4(input: &Vec<String>) -> [Vec<String>; 4] {
    let len = input.len();
    let chunk_size = (len + 3) / 4; // 올림 나눗셈으로 각 Vec의 최소 크기 계산
    let mut result: [Vec<String>; 4] = Default::default();

    for i in 0..4 {
        let start = i * chunk_size;
        let end = usize::min((i + 1) * chunk_size, len);
        result[i] = input[start..end].to_vec();
    }

    result
}
fn div_tensor_vec_4(input: &Vec<Tensor>) -> [Vec<Tensor>; 4] {
    let len = input.len();
    let chunk_size = (len + 3) / 4; // 올림 나눗셈으로 각 Vec의 최소 크기 계산
    let mut result: [Vec<Tensor>; 4] = Default::default();

    for i in 0..4 {
        let start = i * chunk_size;
        let end = usize::min((i + 1) * chunk_size, len);
        result[i] = input[start..end].to_vec();
    }

    result
}
fn save_tensors_to_parquet(
    tensors: &[Tensor],
    shot_num_strs: Vec<String>,
    output_path: &str,
    mode: &str,
    device: &Device,
) -> Result<()> {
    println!("[save_tensors_to_parquet] output_path = {}", output_path);

    let dir_path = Path::new(output_path);
    if !dir_path.exists() {
        println!("Creating directory: {}", dir_path.display());
        std::fs::create_dir_all(&dir_path)?;
    } else if dir_path.is_dir(){
        println!("Directory already exists: {}", dir_path.display());
    } else {
        println!("Path exists but is not a directory: {}", dir_path.display());
        return Err(anyhow::anyhow!("Path exists but is not a directory"));
    }


    let n_rows = tensors
        .iter()
        .map(|t| t.shape().dims()[0])
        .min()
        .unwrap_or(50);
    // 가장 작은 행과 열 찾기
    // let min_cols = tensors.iter()
    //     .map(|t| t.shape().dims()[1])
    //     .min()
    //     .unwrap_or(31400);
    let max_cols = tensors
        .iter()
        .map(|t| t.shape().dims()[1])
        .max()
        .unwrap_or(0);

    println!("max_len is = {}", max_cols);

    let mut fields = vec![
        Field::new("shotnum_id", DataType::Utf8, false),
        Field::new("row_id", DataType::Utf8, false),
        Field::new("len_id", DataType::Utf8, false),
    ];

    for i in 0..max_cols {
        fields.push(Field::new(
            &format!("feature_{}", i),
            DataType::Float32,
            true,
        ));
    }

    let schema = Arc::new(Schema::new(fields));
    let div_shotnum_vec = div_string_vec_4(&shot_num_strs);
    let div_tensor_vec = div_tensor_vec_4(&(tensors.to_vec()));
    let zipped = div_shotnum_vec
        .iter()
        .zip(div_tensor_vec.iter());

    for (idx, (shotnum_vec,tensor_vec)) in zipped.enumerate(){
        let file_name = format!("{}_{}_data.parquet", mode, idx);
        let file_path = format!("{}/{}", output_path, file_name);
        println!("[save_tensors_to_parquet] file_path = {}", file_path);

        let data_file_path = Path::new(&file_path[..]);
        if !data_file_path.exists() {
            println!("Creating file: {}", data_file_path.display());
        } else {
            println!("File already exists: {}", data_file_path.display());
            continue;
        }
    
        let file = File::create(file_path)?;
        let props = WriterProperties::builder()
            .set_compression(parquet::basic::Compression::UNCOMPRESSED)
            .build();
    
        let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props))?;
        assert_eq!(tensor_vec.len(), shotnum_vec.len());


        println!("[save_tensors_to_parquet]: save");
        for (tensor_idx, tensor) in tqdm(tensor_vec.iter()).enumerate() {
            // 텐서를 최소 크기로 자르기
            // let truncated_tensor = truncate_tensor(tensor,  min_cols)?;
            let shot_num = shotnum_vec.get(tensor_idx).unwrap().parse::<usize>()?;
            // println!("shot num is {}", shot_num);

            let tensor_len = tensor.shape().dims()[1];
            // println!("tensor shape = {:?}", tensor.shape());

            let padded_tensor = pad_tensor(tensor, max_cols)?;

            // CPU로 이동하고 데이터 추출
            // let tensor_cpu = truncated_tensor.to_device(&Device::Cpu)?;
            let tensor_cpu = padded_tensor.to_device(&Device::Cpu)?;
            let tensor_data = tensor_cpu.to_vec2::<f32>()?;

            // 각 행별로 레코드 배치 생성
            for row_idx in 0..n_rows {
                // 텐서 ID 및 행 ID 배열 생성
                let shot_num_id_array = StringArray::from(vec![format!("{}", shot_num)]);
                let row_id_array = StringArray::from(vec![format!("row_{}", row_idx)]);
                let len_id_array = StringArray::from(vec![format!("{}", tensor_len)]);
                // 특성 값 배열 생성
                let mut feature_arrays = Vec::with_capacity(max_cols);
                for col_idx in 0..max_cols {
                    let feature_array = Float32Array::from(vec![tensor_data[row_idx][col_idx]]);
                    feature_arrays.push(Arc::new(feature_array) as Arc<dyn arrow::array::Array>);
                }

                // 모든 배열 결합
                let mut arrays = Vec::with_capacity(max_cols + 3);
                arrays.push(Arc::new(shot_num_id_array) as Arc<dyn arrow::array::Array>);
                arrays.push(Arc::new(row_id_array) as Arc<dyn arrow::array::Array>);
                arrays.push(Arc::new(len_id_array) as Arc<dyn arrow::array::Array>);
                arrays.extend(feature_arrays);

                // 레코드 배치 생성 및 기록
                let record_batch = RecordBatch::try_new(schema.clone(), arrays)?;
                writer.write(&record_batch)?;
            }
        }

        // 파일 닫기
        writer.close()?;
    }

    // let shotnum_len = shot_num_strs.len();
    // let div_num = if shotnum_len %4 ==0{
    //     4usize
    // } else  if shotnum_len %3 ==0{
    //     3usize
    // } else{
    //     1usize
    // };
    
    // println!(
    //     "tensor len = {}, shotnum len = {}",
    //     tensors.len(),
    //     shot_num_strs.len()
    // );
    
    Ok(())
}
fn save_tensors_to_bin(
    tensors: &[Tensor],
    shot_num_strs: Vec<String>,
    mode: &str,
    device: &Device,
) -> Result<()> {
    

    let file_path = get_bin_file_path(mode)?;
    let shotnum_vec = shot_num_strs;
    let tensor_vec = tensors.to_vec();
    // for (idx, (shotnum_vec,tensor_vec)) in zipped.enumerate()
    let max_cols = tensors
        .iter()
        .map(|t| t.shape().dims()[1])
        .max()
        .unwrap_or(0);

    let n_rows = tensors
        .iter()
        .map(|t| t.shape().dims()[0])
        .min()
        .unwrap_or(50);

        println!("[save_tensors_to_bin]: save");
        let mut datahash:HashMap<String, Vec<Vec<f32>>> = HashMap::with_capacity(tensor_vec.len());
        let mut lenhash:HashMap<String, usize >= HashMap::with_capacity(tensor_vec.len());
        for (tensor_idx, tensor) in tqdm(tensor_vec.iter()).enumerate() {
            // 텐서를 최소 크기로 자르기
            // let truncated_tensor = truncate_tensor(tensor,  min_cols)?;
            let shot_num = shotnum_vec.get(tensor_idx).unwrap();
            // println!("shot num is {}", shot_num);

            let tensor_len = tensor.shape().dims()[1];
            // println!("tensor shape = {:?}", tensor.shape());

            let padded_tensor = pad_tensor(tensor, max_cols)?;

            // CPU로 이동하고 데이터 추출
            // let tensor_cpu = truncated_tensor.to_device(&Device::Cpu)?;
            let tensor_cpu = padded_tensor.to_device(&Device::Cpu)?;
            let tensor_data = tensor_cpu.to_vec2::<f32>()?;
            datahash.insert(shot_num.clone(), tensor_data);
            lenhash.insert(shot_num.clone(),tensor_len);
        }
        let  bdata = binData{
            data:datahash,
            len:lenhash,
        };
        save_bin(&bdata.data,&bdata.len, &file_path[..])?;
    
    Ok(())
}
// pub struct  SpermWhaleDataset_{
//     data: DataFrame,

fn run_avg_downsample(in_data: Vec<f32>, win_size: usize) -> Result<Vec<f32>> {
    let len = in_data.len();
    if len < win_size {
        return Ok(in_data);
    }
    let new_len = len - (win_size - 1); // Running average keeps the sliding window approach

    let mut result = Vec::new();
    for i in 0..new_len {
        let avg: f32 = in_data[i..i + win_size].iter().sum::<f32>() / (win_size as f32);
        result.push(avg);
    }
    return Ok(result);
}
fn downsample_running_average(data: &Vec<f32>, window_size: usize) -> Result<Vec<f32>>{
    let data_len = data.len();
    if data_len == 0 || window_size == 0 {
        return Ok(Vec::new());
    }

    // let window_size = data_len / target_len;
    let target_len = data_len/ window_size; 
    let mut result = Vec::with_capacity(target_len);

    for i in 0..target_len {
        let start = i * window_size;
        let end = if i == target_len - 1 {
            data_len // 마지막은 남은 모든 데이터를 포함
        } else {
            (i + 1) * window_size
        };
        let window = &data[start..end];
        let sum: f32 = window.iter().sum();
        let avg = sum / (window.len() as f32);
        result.push(avg);
    }

    Ok(result)
}
pub struct SCDataset_ {
    data_map: HashMap<String, Tensor>,
    length_map: HashMap<String, usize>,
    shotnum_vec: Vec<String>,
    shotnum_indices: Vec<usize>,
    data_window: usize,
    batch_size: usize,
    index_size: usize,
    drop_len: usize,
    // data_vec: Vec<Tensor>,
    // length_vec: Vec<usize>,
    max_length: usize,
    device: Device,
}
#[derive(Clone)]
// pub struct SCDataset(Rc<SCDataset_>);
pub struct SCDataset(Arc<SCDataset_>);

impl AsRef<SCDataset> for SCDataset {
    fn as_ref(&self) -> &SCDataset {
        self
    }
}

impl std::ops::Deref for SCDataset {
    type Target = SCDataset_;
    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}

impl SCDataset {
    pub fn new(
        data_map: &HashMap<String, Tensor>,
        length_map: &HashMap<String, usize>,
        data_window: usize,
        batch_size: usize,
        device: Device,
    ) -> Self {
        println!("[SCDataset][new] data_map len = {}", data_map.len());
        let shotnum_vec: Vec<String> = data_map.iter().map(|(key, _)| String::from(key)).collect();
        let mut shotnum_indices: Vec<usize> = (0..shotnum_vec.len()).collect();

        let mut rng = thread_rng();
        shotnum_indices.shuffle(&mut rng);

        // // data_vec, length_vec code
        // let mut data_vec: Vec<Tensor> = Vec::new();
        // let mut length_vec: Vec<usize> = Vec::new();
        // for shotnum in &shotnum_vec {
        //     let tensor_o = data_map.get(shotnum); //.ok_or_else(|| anyhow::anyhow!("shotnum is not found at data_map"))?;
        //     match tensor_o {
        //         Some(tensor) => {
        //             data_vec.push(tensor.clone());
        //         }
        //         None => {
        //             println!("shotnum is not found at data_map");
        //         }
        //     }
        //     let len_o = length_map.get(shotnum);
        //     match len_o {
        //         Some(len) => {
        //             length_vec.push(len.clone());
        //         } // if len > &max_len{ max_len = *len;}}
        //         None => {
        //             println!("len is not found at data_map");
        //         }
        //     }
        // }
        // println!("[SCDataset][new] make data_vec= {}, length_vec = {} done", data_vec.len(),length_vec.len());

        let mut max_len: usize = 0;
        match data_map.get(shotnum_vec[0].as_str()) {
            Some(tensor) => max_len = tensor.dims()[1],
            None => {
                println!("getting is data_vec[1] length is failed");
            }
        }
        println!("max_len = {}", max_len);
        let index_size_org = max_len - (data_window - 1);
        let drop_len = index_size_org % batch_size; // 284 / 32 = 8...28
        let index_size = index_size_org - drop_len;
        println!("index_size_org = {}", index_size_org);
        println!("index_size = {}", index_size);
        println!("drop_len = {}", drop_len);

        let dataset_ = SCDataset_ {
            data_map: data_map.clone(),
            length_map: length_map.clone(),
            shotnum_vec: shotnum_vec,
            shotnum_indices: shotnum_indices,
            data_window: data_window,
            batch_size: batch_size,
            drop_len: drop_len,
            index_size: index_size,
            // data_vec: data_vec,
            // length_vec: length_vec,
            max_length: max_len,
            device: device,
        };
        Self(Arc::new(dataset_))
    }
    pub fn shuffle(&self) {
        
    }
    pub fn len(&self) -> usize {
        // let max_len = self.lengt
        // println!("max_len : {}", self.max_length);
        // println!("data_window : {}", self.data_window);
        // println!("length_map : {}", self.length_map.len());

        let len = (self.get_index_size()) * self.length_map.len();
        // let len: usize = self.length_map.values().sum::<usize>() - (self.data_window - 1)* self.length_map.len();
        return len;
    }
    pub fn get_max_len(&self)->usize{
        self.max_length
    }
    pub fn get_index_size(&self)->usize{
        // self.max_length - (self.data_window - 1) - self.drop_len
        self.index_size
    }
    pub fn get_num_of_shots(&self)->usize{
        self.length_map.len()
    }
    pub fn get_device(&self) -> &Device {
        &self.device
    }
    pub fn get_shotnum_vec(&self) -> &Vec<String> {
        &self.shotnum_vec
    }
    pub fn get_idx(&self, idx: usize) -> anyhow::Result<(usize, usize)> {
        let elem_len = self.max_length;

        if elem_len <= 0 {
            anyhow::bail!("[get_idx]elem_len is negative or zero");
        }
        if idx < 0 {
            anyhow::bail!("[get_idx]idx is negative ");
        }
        let main_idx: usize = (idx / elem_len) as usize;
        let sub_idx: usize = (idx % elem_len) as usize;
        Ok((main_idx, sub_idx))
    }
    pub fn is_empty(&self) -> bool {
        self.data_map.is_empty()
    }
    pub fn get_tensor(&self, idx: usize) -> anyhow::Result<(&Tensor, usize)> {
        let shotnum = self
            .shotnum_vec
            .get(idx)
            .ok_or_else(|| anyhow::anyhow!("Index out of bounds"))?;
        let tensor = self
            .data_map
            .get(shotnum)
            .ok_or_else(|| anyhow::anyhow!("Tensor not found"))?;
        let len = self
            .length_map
            .get(shotnum)
            .ok_or_else(|| anyhow::anyhow!("Length not found"))?;
        Ok((tensor, len.clone()))
    }
    pub fn get_tensor_random(&self, idx: usize) -> anyhow::Result<(&Tensor, usize)> {
        let ridx = self.shotnum_indices.get(idx)
            .ok_or_else(|| anyhow::anyhow!("Index out of bounds"))?;
        let shotnum = self
            .shotnum_vec
            .get(*ridx)
            .ok_or_else(|| anyhow::anyhow!("Index out of bounds"))?;
        let tensor = self
            .data_map
            .get(shotnum)
            .ok_or_else(|| anyhow::anyhow!("Tensor not found"))?;
        let len = self
            .length_map
            .get(shotnum)
            .ok_or_else(|| anyhow::anyhow!("Length not found"))?;
        Ok((tensor, len.clone()))
    }
    pub fn get_tensor_by_shotnum(&self, shotnum: &str) -> anyhow::Result<(&Tensor, usize)> {
        let tensor = self
            .data_map
            .get(shotnum)
            .ok_or_else(|| anyhow::anyhow!("Tensor not found"))?;
        let len = self
            .length_map
            .get(shotnum)
            .ok_or_else(|| anyhow::anyhow!("Length not found"))?;
        Ok((tensor, len.clone()))
    }

}

fn read_hdf5_data(
    mode: &str,
    scale_factor: usize,
    device: &Device,
) -> anyhow::Result<(Vec<Tensor>, Vec<String>)> {
    println!("[read_hdf5_data] mode = {}", mode);
    let mut scaler = MinMaxScaler::new((-1.0, 1.0), device);

    let path_str = format!("./dataset/{}/{}_data.hdf5", mode, mode);
    println!("{:?}", path_str);

    let file = hFile::open(path_str).unwrap();
    let root_group = file.group(&format!("/{}", mode)[..])?;
    let mut shot_number_iter = root_group.member_names()?.into_iter();
    let shotnum_vec = shot_number_iter.clone().collect::<Vec<String>>();

    let mut tot_tensor_vec = Vec::<Tensor>::new();
    // for shotnum_str in shot_number_iter {
    println!("Loading HDF5 data :");
    for shotnum_str in tqdm(shotnum_vec.iter()) {
        let shot_group = root_group.group(&shotnum_str[..])?;
        let member = shot_group.member_names()?;

        let mut tensor_vec = Vec::<Tensor>::new();

        for tag in TAG_NAMES.iter() {
            if let Ok(tag_group) = shot_group.group(tag) {
                if let Ok(dataset) = tag_group.dataset("val") {
                    
                    let data = dataset.read_raw::<f32>().unwrap();
                    // println!("data shape ={}", data.len());
                    // let data_run_avg = run_avg_downsample(data.clone(), scale_factor)?;
                    let data_run_avg = downsample_running_average(&data, scale_factor)?;
                    
                    // println!("data_run_avg shape ={}", data_run_avg.len());
                    let tensor = Tensor::from_vec(
                        data_run_avg.clone(),
                        (1, data_run_avg.len()),
                        device,
                    )?;
                    tensor_vec.push(tensor);
                }
            }
        }

        let concat_tesnor = Tensor::cat(&tensor_vec, 0)?;
        // println!("{:?}", concat_tesnor.shape());
        scaler.partial_fit(&concat_tesnor)?;
        tot_tensor_vec.push(concat_tesnor);
    }
    // let mut scaled_tot_tensor_vec = Vec::<Tensor>::new();
    let scaled_tot_tensor_vec= tot_tensor_vec
        .iter()
        .map(|tensor| {
            // let scaled_tensor = scaler.transform(tensor).unwrap();
            // scaled_tot_tensor_vec.push(scaled_tensor);
           scaler.transform(tensor).unwrap()
        })
        .collect::<Vec<_>>();
    println!("tot_tensor_vec[0]: {}", tot_tensor_vec[0]);
    println!("scaled_tot_tensor_vec[0]: {}", scaled_tot_tensor_vec[0]);
    println!(
        "scaled_tot_tensor_vec len: {:?}",
        scaled_tot_tensor_vec.len()
    );
    println!(
        "scaled_tot_tensor_vec shape: {:?}",
        scaled_tot_tensor_vec[0].shape()
    );
    // scaled_tot_tensor_vec,shotnum_vec,
    Ok((scaled_tot_tensor_vec, shotnum_vec))
}
use std::path::PathBuf;
use std::{fs, io};

fn get_file_paths_in_folder(folder_path: &str) -> Result<Vec<String>> {
    let mut file_vec = Vec::new();
    let entries = fs::read_dir(folder_path)?;

    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            if let Some(extension) = path.extension() {
                if extension == "parquet" {
                    if let Some(path_str) = path.to_str() {
                        file_vec.push(path_str.to_string());
                    }
                }
            }
        }
    }

    Ok(file_vec)
}
pub struct SCDatasetBuilder<'a> {
    data_: Option<HashMap<String, Tensor>>,
    length_: Option<HashMap<String, usize>>,
    shotnum_: Option<Vec<String>>,
    mode: String,
    device: &'a Device,
    data_window: usize,
    batch_size: usize,
}
impl<'a> SCDatasetBuilder<'a> {
    pub fn new(mode: &str, device: &'a Device, data_window: usize,batch_size: usize) -> Self {
            Self {
                data_: None,
                length_: None,
                shotnum_: None,
                mode: String::from(mode),
                device,
                data_window: data_window,
                batch_size: batch_size,
            }
        }
    // pub fn get_device(&self) -> &Device {
    //     &self.device
    // }
    pub fn load_data(mut self) -> Self {
        println!("[load_data] mode = {}", self.mode);
        // let dir_path = format!("./parquet/{}_data",self.mode);
        // let file_name = format!("{}_{}_data.parquet", self.mode, 3usize);
        // let file_path = format!("{}/{}", dir_path, file_name);

        
        // let data_file_path = Path::new(&file_path[..]);
        let dir_path = match get_bin_dir(&self.mode[..]){
            Ok(path) => {
                println!("Success get get_bin_dirs {}", path.clone());
                path
            }
            Err(e) => {
                panic!("Error get get_bin_dirs {}", e);
            }
        };
        let file_path = match get_bin_file_path(&self.mode[..]){
            Ok(path) => {
                println!("Success get get_bin_file_path {}", path.clone());
                path
            }
            Err(e) => {
                panic!("Error get get_bin_file_path {}", e);
            }
        };
        let data_file_path = Path::new(&file_path[..]);
        println!("data_file_path = {:?}", data_file_path);
        let mut shotnum_vec: Vec<String> = Vec::new();
        let mut scaled_tot_tensor_vec: Vec<Tensor> = Vec::new();
        if !data_file_path.exists() {
            // let (scaled_tot_tensor_vec_, shotnum_vec_) = read_hdf5_data(&self.mode[..],100,self.device)?;
            let (scaled_tot_tensor_vec_, shotnum_vec_) =
                match read_hdf5_data(&self.mode[..], 100, &Device::Cpu) {
                    Ok((scaled_tot_tensor_vec_, shotnum_vec_)) => {
                        (scaled_tot_tensor_vec_, shotnum_vec_)
                    }
                    Err(e) => {
                        // println!("Error reading HDF5 data: {}", e);
                        // return self;
                        panic!("Error reading HDF5 data: {}", e);
                    }
                };
            // save_tensors_to_parquet(&scaled_tot_tensor_vec_,shotnum_vec_.clone(), &format!("{}_data.parquet",self.mode)[..])?;
            // match save_tensors_to_parquet(
            //     &scaled_tot_tensor_vec_,
            //     shotnum_vec_.clone(),
            //     &format!("./parquet/{}_data", self.mode)[..],
            //     &self.mode[..],
            //     &Device::Cpu,
            //     // &format!("{}_data.parquet", self.mode)[..],
            // ) {
            //     Ok(()) => {
            //         println!("Success saving tensor to {}_data.parquet", self.mode);
            //     }
            //     Err(e) => {
            //         panic!("Error saving tensors to  {}_data.parquet {}", self.mode, e);
            //     }
            // };
            match save_tensors_to_bin(
                &scaled_tot_tensor_vec_,
                shotnum_vec_.clone(),
                &self.mode[..],
                &Device::Cpu,
                // &format!("{}_data.parquet", self.mode)[..],
            ) {
                Ok(()) => {
                    println!("Success saving tensor to {}_data.parquet", self.mode);
                }
                Err(e) => {
                    panic!("Error saving tensors to  {}_data.parquet {}", self.mode, e);
                }
            };
            shotnum_vec = shotnum_vec_;
            scaled_tot_tensor_vec = scaled_tot_tensor_vec_;
        }
        // let dir_path = match get_bin_dir(&self.mode[..]){
        //     Ok(path) => {
        //         println!("Success get get_bin_dirs {}", path.clone());
        //         path
        //     }
        //     Err(e) => {
        //         panic!("Error get get_bin_dirs {}", e);
        //     }
        // };
        let file_path_vec 
            = match get_binfile_paths_in_folder(&dir_path[..]){
                Ok(file_vec) => {
                    println!("Success get file paths ./bincode/{}_data", self.mode);
                    file_vec
                }
                Err(e) => {
                    panic!("Error get file paths ./bincode/{}_data {}", self.mode, e);
                }
            };
        
        let mut out_tensor_map : HashMap<String, Tensor> = HashMap::new();
        let mut out_len_map : HashMap<String, usize> = HashMap::new();
        for f_path in file_path_vec {
            println!("[SCDatasetBuilder][load_data]{}", f_path);
        // let (tensor_map2, len_map) = read_parquet_to_tensors(&format!("{}_data.parquet", MODES[1])[..])?;
            let t5 = std::time::Instant::now();
            // let (tensor_map2, len_map) =
            //     // match read_parquet_to_tensors(&format!("{}_data.parquet", self.mode)[..]) {
                    
            //     match read_parquet_to_tensors(&f_path[..],&self.device) {

            //         Ok((tensor_map2, len_map)) => (tensor_map2, len_map),
            //         Err(e) => {
            //             // println!("Error reading HDF5 data: {}", e);
            //             // return self;
            //             panic!("Error reading {}_data.parquet data: {}", self.mode, e);
            //         }
  
            //     };
            let (tensor_map2, len_map) =
              
                match read_bin_to_tensors(&f_path[..],&self.device) {

                    Ok((tensor_map2, len_map)) => (tensor_map2, len_map),
                    Err(e) => {
                        // println!("Error reading HDF5 data: {}", e);
                        // return self;
                        panic!("Error reading {}_data.parquet data: {}", self.mode, e);
                    }
  
                };
            let dt5 = t5.elapsed();
            println!( "read_parquet_to_tensors:{} t5 ({}s)",f_path,dt5.as_secs_f64(),);
                out_tensor_map.extend(tensor_map2);
                out_len_map.extend(len_map);

        }

        self.data_ = Some(out_tensor_map);
        self.length_ = Some(out_len_map);
        self.shotnum_ = Some(shotnum_vec);
        self
    }
    pub fn build(&self) -> SCDataset {
        // let t6 = std::time::Instant::now();
        if let Some(data_) = &self.data_ {
            
            if let Some(length_) = &self.length_ {
                SCDataset::new(data_, length_, self.data_window,self.batch_size, self.device.clone())
            } else {
                panic!("length hashmap is not set SCDatasetBuilder.");
            }

        } else {
            panic!("data hashmap is not set SCDatasetBuilder.");
        }
        // let dt6 = t6.elapsed();
        // println!( "SCDataset::new() t6 ({}s)",dt6.as_secs_f64(),);
    }
}
pub struct SCDatasetIter<'a> {
    dataset: SCDataset,
    remaining_indices: Vec<usize>,
    input_window: usize,
    output_window: usize,
    device: &'a Device,
}
impl<'a> SCDatasetIter<'a> {
    pub fn new(
        dataset: SCDataset,
        shuffle: bool,
        input_window: usize,
        output_window: usize,
        device: &'a Device
    ) -> Self {
        // let mut remaining_indices: Vec<usize> = (0..dataset.len()).rev().collect::<Vec<_>>();
        let mut remaining_indices_rev: Vec<usize> = Vec::with_capacity(dataset.len());
        // let ind_time = std::time::Instant::now();
        for main_idx in (0..dataset.get_num_of_shots()){
            for sub_idx in (0..dataset.get_index_size()){
                let idx = main_idx* dataset.get_max_len() + sub_idx;
                remaining_indices_rev.push(idx)
            }
        }
        let mut remaining_indices: Vec<_> = remaining_indices_rev.iter().rev().cloned().collect();
        // let ind_dt = ind_time.elapsed();
        // println!( "indexing time  ({}s)",ind_dt.as_secs_f64(),);
        // println!("remaining_indices = {}",remaining_indices.len());
        if shuffle {
            let mut rng = thread_rng();
            remaining_indices.shuffle(&mut rng);
        }

        Self {
            dataset,
            remaining_indices,
            input_window,
            output_window,
            device,
        }
    }
    pub fn create_xy_seq(&self, i: usize, tensor: &Tensor) -> anyhow::Result<(Tensor, Tensor)> {
        let data_len = tensor.dims()[1] as usize;
        let chan_dim = tensor.dims()[0] as usize;

        // println!("chan_dim /data_len/i = {}/{}/{}", chan_dim, data_len, i);
        let shape = (chan_dim, 1);
        let mut x = Tensor::zeros(shape, candle_core::DType::F32, self.device)?;
        let mut y = Tensor::zeros(shape, candle_core::DType::F32, self.device)?;

        if i >= data_len {
            anyhow::bail!(format!(
                "creae_xy_seq has wrong index i: {}[data_len: {}]",
                i, data_len
            ));
        }
        // println!("[create_xy_seq] tensor = {}",tensor);
        let x_slice = tensor.narrow(1, i, 1)?; //.contiguous()?;
                                                    // (0..chan_dim, i..(i+in_win)));
        // println!(
        //     "debug1 data_len/in/out_len{}{}{}",
        //     data_len, in_win, out_win
        // );
        // println!("tesnsor shape = {:?}", tensor.shape());
        // println!("x_slice shape {:?}", x_slice.shape());
        let x_out = x.slice_assign(&[0..chan_dim, 0..1], &x_slice)?;

        let y_slice = tensor.narrow(1, i, 1)?.contiguous()?;
        // (0..chan_dim, i..(i+in_win+out_win))?;
        // println!("y_slice shape {:?}", y_slice.shape());
        let y_out = y.slice_assign(&[0..chan_dim, 0..1], &y_slice)?;
        // let x = tensor.slice(0, sub_idx as i64, self.dataset.data_window as i64);
        // let y = tensor.slice(0, (sub_idx + 1) as i64, 1);
        // println!("[create_xy_seq] x = {}",x);
        // println!("[create_xy_seq] y= {}",y);
        // println!("[create_xy_seq] x_out = {}",x_out);
        // println!("[create_xy_seq] y_out= {}",y_out);
        Ok((x_out, y_out))
    }
}

impl<'a> Iterator for SCDatasetIter<'a> {
    type Item = candle_core::Result<(Tensor, Tensor)>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(idx) = self.remaining_indices.pop() {
            let (main_idx, sub_idx) = self.dataset.get_idx(idx).unwrap();
            let (tensor, len) = self.dataset.get_tensor(main_idx).unwrap();
            // let (tensor, len) = self.dataset.get_tensor_random(main_idx).unwrap();
            let tensor = tensor
                .clone()
                .to_device(self.device)
                .unwrap();
            // self.creae_xy_seq(&self,i:usize, tensor: &Tensor )
            match self.create_xy_seq(sub_idx, &tensor) {
                Ok((x, y)) => Some(Ok((x, y))),
                Err(e) => {
                    println!("create_xy_seq if fail at index {}:{}", sub_idx, e);
                    None
                }
            }
            // Some((main_idx, sub_idx, tensor, len))
        } else {
            None
        }
    }
}



pub type SCDataBatcher<'a> = Batcher<IterResult2<SCDatasetIter<'a>>>;

pub struct SCDataLoader<'a> {
    dataset: SCDataset,
    batch_size: usize,
    shuffle: bool,
    drop_last: bool,
    input_window: usize,
    output_window: usize,
    device: &'a Device,
}

impl<'a> SCDataLoader<'a> {
    pub fn new(
        dataset: SCDataset,
        batch_size: usize,
        shuffle: bool,
        drop_last: bool,
        input_window: usize,
        output_window: usize,
        device: &'a Device,

    ) -> Self {
        Self {
            dataset,
            batch_size,
            shuffle,
            drop_last,
            input_window,
            output_window,
            device: device,
        }
    }
    pub fn batcher(&self) -> SCDataBatcher {
        //  let t1 = std::time::Instant::now();
        let iter = SCDatasetIter::new(
            self.dataset.clone(),
            self.shuffle,
            self.input_window,
            self.output_window,
            self.device,
        );
        // let dt1 = t1.elapsed();
        // println!( "[SCDataLoader][batcher][SCDatasetIter::new]time  ({}s)",dt1.as_secs_f64(),);
        //  let t2 = std::time::Instant::now();
        let b = Batcher::new_r2(iter)
            .batch_size(self.batch_size)
            .return_last_incomplete_batch(!self.drop_last);
        // let dt2 = t2.elapsed();
        // println!( "[SCDataLoader][batcher][Batcher::new_r2]time  ({}s)",dt2.as_secs_f64(),);
        return  b
            
    }
    pub fn len(&self) -> usize {
        if self.drop_last {
            self.batcher().count()
        } else {
            // let t1 = std::time::Instant::now();
            let mut batcher = self.batcher();
            let mut count = 0_usize;
            while let Some(Ok(_el)) = batcher.next() {
                count += 1;
            }
            // let dt1 = t1.elapsed();
            // println!( "[SCDataLoader][len]time  ({}s)",dt1.as_secs_f64(),);
            count
        }
         
    }
    pub fn is_empty(&self) -> bool {
        (self.dataset.len() < self.batch_size) && (self.drop_last)
    }
}
