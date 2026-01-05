
use std::collections::HashMap;
use crate::database::{Db,RingBuff,Item};
use crate::process_variable::{PV,PVContext,PVValue};
use plotly::layout::WaterfallMode;
use tokio::sync::{mpsc,broadcast,watch};
use tokio::task::JoinError;
use std::sync::{Arc, Mutex};
use crate::shutdown::Shutdown;
use tokio::time::Instant;

use tokio::time::{self, Duration};
use tokio::time::{sleep};
use tracing::{debug, error, info, instrument};

pub type Error = Box<dyn std::error::Error + Send + Sync>;
pub type Result<T> = std::result::Result<T, Error>;

use chrono::{DateTime, Utc};
use crate::dataset::{SCDataLoader,SCDatasetBuilder};
use crate::dataset::{SCShotDataLoader};
use crate::dataset;

// use anyhow::{anyhow, Result};
use candle_core::{Device, Result as cResult, Tensor, D};
use candle_datasets::{batcher::IterResult2, Batcher};

use tokio::net::TcpStream;
use tokio::io::{AsyncWriteExt, AsyncReadExt};

use crate::config::Config;

use tqdm::pbar;
use crate::dataset::MinMaxScaler;
#[derive(Debug)]
pub struct OUT_TX {
    pv:PV,
    outdata_tx: mpsc::Sender<PVValue>,
}
impl OUT_TX{
    pub async fn send(&mut self,maybe_value: Option<f64>)-> std::result::Result<(),PublisherError>
    {
        let outdata_tx = self.outdata_tx.clone();
        let pv = self.pv.clone();
        let ret = match maybe_value{
            Some(value)=>{
                let handle = tokio::spawn(async move{
                    let pvvalue = PVValue{
                        key: pv.key,
                        pvname: pv.pvname,
                        time_stamp: Some(Utc::now()),
                        wr:pv.wr,
                        enable_ca: pv.enable_ca,
                        value: Some(value),
                        buff_size: pv.buff_size,
                        vec_size: pv.vec_size,
                        pred_size: pv.pred_size,
                    };
                    outdata_tx.send(pvvalue).await
                });
                let ret_val= handle.await?;
                ret_val?
            }
            None => {
                // error!("dbpop is failed. so skip the sending");
            }
        };
        Ok(ret)
    }
}

use std::fmt;
#[derive(Debug)]
pub enum PublisherError{
    SendError,
    JoinError,
}

impl std::error::Error for PublisherError {}

impl fmt::Display for PublisherError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result{
        match self {
            PublisherError::SendError => write!(f, "Send Error"),
            PublisherError::JoinError => write!(f, "Joint Error"),
        }
    }
}
impl From<mpsc::error::SendError<PVValue>> for PublisherError{
    fn from(_: mpsc::error::SendError<PVValue>) -> Self{
        PublisherError::SendError
    } 
}
impl From<JoinError> for PublisherError{
    fn from(_: JoinError) -> Self{
        PublisherError::JoinError
    } 
}

#[derive(Debug)]
pub struct Publisher {
     db: Db,
    // notify_shutdown: broadcast::Sender<()>,
    shutdown: Shutdown,
    shutdown_complete_tx: mpsc::Sender<()>,
    connection: OUT_TX,
    startstop_tx: watch::Sender<bool>,
    pv:PV, 
    config: Config, 
    device: Device,
    index: usize,
}

impl Publisher {
    pub fn new(db: Db, shutdown: Shutdown,
            shutdown_complete_tx: mpsc::Sender<()>,
            pv:&PV,
            outdata_tx: mpsc::Sender<PVValue>,
            startstop_tx: watch::Sender<bool>,
            config: Config,
            device: Device,
            
        )->Publisher
    {
        
        let keys = vec![
            "PF1U_CURRENT","PF2U_CURRENT","PF3U_CURRENT","PF3L_CURRENT",
            "PF4U_CURRENT","PF4L_CURRENT","PF5U_CURRENT","PF5L_CURRENT",
            "PF6U_CURRENT","PF6L_CURRENT","PF7U_CURRENT",
            "PF1L_TEMP_IN","PF1L_TEMP_OUT1","PF1L_TEMP_OUT2","PF1L_TEMP_OUT3",
            "PF1L_TEMP_OUT4","PF1L_TEMP_OUT5","PF1L_TEMP_OUT6",
            "PF1U_TEMP_IN","PF1U_TEMP_OUT1","PF1U_TEMP_OUT2","PF1U_TEMP_OUT3",
            "PF1U_TEMP_OUT4","PF1U_TEMP_OUT5","PF1U_TEMP_OUT6",
            "PF1U_PRES_IN","PF1L_PRES_IN","PF2U_PRES_IN","PF2L_PRES_IN",
            "PF3U_PRES_IN","PF3L_PRES_IN","PF4U_PRES_IN","PF4L_PRES_IN",
            "PF5U_PRES_IN","PF5L_PRES_IN","PF6_PRES_IN","PF7_PRES_IN","PF_PRES_OUT",
            "PF1U_FLOW_IN","PF1L_FLOW_IN","PF2U_FLOW_IN","PF2L_FLOW_IN",
            "PF3U_FLOW_IN","PF3L_FLOW_IN","PF4U_FLOW_IN","PF4L_FLOW_IN",
            "PF5U_FLOW_IN","PF5L_FLOW_IN","PF6_FLOW_IN","PF7_FLOW_IN",
   
        ];
        
        let index_hash: HashMap<String, usize> = keys.iter().enumerate()
            .map(|(i, key)| (key.to_string(), i))
            .collect();
        let index = index_hash.get(&pv.key)
            .expect(&format!("Key {} not found in index_hash", pv.key));
        let idx = *index;
        Publisher{
            db,
            // notify_shutdown,
            shutdown,
            shutdown_complete_tx,
            connection: OUT_TX{pv:pv.clone(), outdata_tx},
            startstop_tx,
            pv: pv.clone(),
            config,
            device,
            index:idx,
        }
    }
    pub  fn dataset_read(&mut self,pv:&PV,db:Db,input_tensor: &Tensor)-> Option<f64>{
        
            // if pv.enable_ca == false{continue;}

            // if pv.wr  &  WRITE_BIN_CODE  == 0 {continue;}

        let key  = pv.key.clone();
        let value = match db.get_front(&key[..]){
            None => {
                error!("Publisher][{}] db pop error",key);
                None
            },
            Some(item) => {
                // println!("[dataset_reader][{}] value: {}/ time: {:?}",key, item.value, item.time_stamp);
                Some(item.value)
            },
        };
        value
    }

    pub async fn run(&mut self) -> Result<()> {
        let data_window = self.config.input_window + self.config.output_window;
        // let test_dataset = SCDatasetBuilder::new("test",
        //      &self.device, data_window,self.config.batch_size)
        //     .load_data()
        //     .build();


        let test_builder = SCDatasetBuilder::new(
            "test", 
            &self.device, 
            data_window,
            self.config.batch_size,
            (self.config.rng_seed) as u64,
            self.config.sampling_rate,
            0, 4).load_data();
        // let mut shotnum_vec = vec![];
        let test_dataset_vec = test_builder.build_shotdata();


        let dev = self.device.clone();
  
        let out_win = self.config.output_window;
        // let mut test_dataloader_vec = Vec::with_capacity(test_dataset_vec.len());
        // for test_ds in test_dataset_vec.iter(){
        //     let shot_num = test_ds.get_shotnum();
        //     shotnum_vec.push(shot_num.clone());
        //     // let test_dataset = test_ds.clone();
        //     let test_loader = SCShotDataLoader::new(
        //         test_ds.clone(),
        //         1,
        //         false,
        //         true,
        //         self.config.input_window,
        //         self.config.output_window,
        //         &self.device,
        //     );
        //     test_dataloader_vec.push(test_loader);
        // }
        // let mut test_batcher = test_loader.batcher();

        // let mut shutdown = Shutdown::new(self.notify_shutdown.subscribe());
        let mut scaler_tmp = MinMaxScaler::new((-1.0, 1.0), &self.device);
        let mut scaler = scaler_tmp.load("test", &self.device)?;

        let mut shot_idx = 0;
        let mut batch_idx = 0;

        // let mut stream = TcpStream::connect("127.0.0.1:9000").await?;
        while !self.shutdown.is_shutdown() {
            let start = Instant::now();
            let pv = self.pv.clone();
            let mut value = None;
            let mut input_tensor_o= None;
            // stream.write_all(b"start").await?;
            // if let Some(Ok((input_batch, target_batch))) = test_batcher.next()
            // let test_dl = test_dataloader_vec.get(shot_idx).unwrap();
            let test_dataset = test_dataset_vec.get(shot_idx).unwrap();
             

            // let mut test_batcher: Batcher<IterResult2<dataset::SCShotDatasetIter<'_>>>= test_dl.batcher();
            // let mut test_batch_vec = vec![];
            // while let Some(Ok((ib, tb))) = test_batcher.next() {
            //     test_batch_vec.push((ib, tb));
            // }
            // let tt = test_batch_vec[batch_idx].clone();
            // if test_batch_vec.len() > batch_idx
             if (test_dataset.len() - out_win) > batch_idx {
                if (self.index == 15){
                    if batch_idx == 0{ 
                        self.startstop_tx.send(true)?;
                    }
                }
                let (input_batch, len) = test_dataset.get_tensor().unwrap().clone();
                // println!("input_batch: {:?}, target_batch: {:?}", input_batch.shape(), target_batch.shape());
                // let input_batch = data_tensor

            
        
                let input_batch2 = input_batch.to_device(&self.device).unwrap();
                // println!("input_batch2: {:?}", input_batch2.shape());
                
                // let input_batch2  = input_batch2.squeeze(0).unwrap();
                // println!("input_batch2: {:?}", input_batch2.shape());
                let scaled_input_tensor = scaler.inverse_transform(&input_batch2)?;
                // println!("scaled_input_tensor: {:?}", scaled_input_tensor.shape());
                // let scaled_input_tensor  = scaled_input_tensor.squeeze(1).unwrap();
                // println!("scaled_input_tensor: {:?}", scaled_input_tensor.shape());
                // let input = input_batch2.index_select(0, self.index as u64).unwrap();
                
                let input_tensor = scaled_input_tensor.get(self.index).unwrap();
                let input_tensor = input_tensor.get(batch_idx).unwrap();
              
                // println!("input_tensor: {:?}", input_tensor.shape());
                input_tensor_o = Some(input_tensor.clone());
                // let input_tensor = input_tensor.to_dtype(DType::F32).unwrap();
                // value = self.dataset_read(&pv,self.db.clone(),&input_tensor);
                // let v = match value{
                //     Some(v) => v,
                //     None => continue,
                // };
                batch_idx += 1;

            } else {
                // error!("Failed to get next batch from train_batcher");
                batch_idx = 0;
                shot_idx += 1;
                if (self.index == 15){
                    self.startstop_tx.send(false)?;
                }
                println!("shot_idx increased to {} batch_idx = {}", shot_idx,batch_idx);
            }
            let input_tensor = match input_tensor_o {
                Some(t) => t,
                None => {
                    error!("Failed to get input tensor");
                    continue;
                },
            };
            // let len = input_tensor.shape().dims()[0];
            // println!("input_tensor len: {:?}", len);
            // for it in 0..len {
            let valf32:f32 = input_tensor.to_scalar().unwrap();
            let val = valf32 as f64;
                //   println!("input_tensor[{}]: {}", pv.key,val);
                // let itv = input_tensor.get().unwrap().to_scalar::<f64>().unwrap();
                // value = Some(itv);
                value = Some(val);
                let ret_value = tokio::select! {
                    res = self.connection.send(value) => res,
                    _ = self.shutdown.recv() => {
                        info!("[Publisher]shutdown");
                        // If a shutdown signal is received, return from `run`.
                        // This will result in the task terminating.
                        return Ok(());
                    }
                };
                time::sleep(Duration::from_millis(1)).await;

            // }

                // let key  = &self.pv.key;
                // let value = match self.db.get_front(&key[..]){
                //     None => {
                //         error!("Publisher][{}] db pop error",key);
                //         None
                //     },
                //     Some(item) => {
                //         // println!("[DB_Reader][{}] value: {}/ time: {:?}",key, item.value, item.time_stamp);
                //         Some(item.value)
                //     },
                // };

            // time::sleep(Duration::from_millis(1000)).await;
            let dt = start.elapsed(); // 시작 시점부터 현재까지의 경과 시간 계산
            println!("[Publisher] dt: {:?}", dt);
            let dur = Duration::from_millis(1000).saturating_sub(dt); // 1초에서 경과 시간을 뺌
            // let dur = Duration::from_millis(200).saturating_sub(dt); // 1초에서 경과 시간을 뺌
            // let dur = Duration::from_micros(100).saturating_sub(dt); // 1초에서 경과 시간을 뺌
            sleep(dur).await; // 

        }
        return Ok(())
    }
    // pub async fn run2(&mut self) -> Result<()> {
    //     let data_window = self.config.input_window + self.config.output_window;
    //     let test_dataset = SCDatasetBuilder::new("test",
    //          &self.device, data_window,self.config.batch_size)
    //         .load_data()
    //         .build();
    //     let dev = self.device.clone();
    //     let test_loader = SCDataLoader::new(
    //             test_dataset.clone(),
    //            1,
    //             false,
    //             true,
    //             self.config.input_window,
    //             self.config.output_window,
    //             &dev,
    //         );   
    //     let mut test_batcher = test_loader.batcher();

    //     // let mut shutdown = Shutdown::new(self.notify_shutdown.subscribe());
    //     let mut scaler_tmp = MinMaxScaler::new((-1.0, 1.0), &self.device);
    //     let mut scaler = scaler_tmp.load("test", &self.device)?;

    //     let mut stream = TcpStream::connect("127.0.0.1:9000").await?;
    //     while !self.shutdown.is_shutdown() {
    //         let start = Instant::now();
    //         let pv = self.pv.clone();
    //         let mut value = None;
    //         let mut input_tensor_o= None;
    //         stream.write_all(b"start").await?;
    //         if let Some(Ok((input_batch, target_batch))) = test_batcher.next() {
    //             // println!("input_batch: {:?}, target_batch: {:?}", input_batch.shape(), target_batch.shape());

    //             let input_batch2 = input_batch.to_device(&self.device).unwrap();
    //             // println!("input_batch2: {:?}", input_batch2.shape());
    //             let target_batch2 = target_batch.to_device(&self.device).unwrap();
    //             let input_batch2  = input_batch2.squeeze(0).unwrap();
    //             // println!("input_batch2: {:?}", input_batch2.shape());
    //             let scaled_input_tensor = scaler.inverse_transform(&input_batch2)?;
    //             let scaled_input_tensor  = scaled_input_tensor.squeeze(1).unwrap();
    //             // println!("scaled_input_tensor: {:?}", scaled_input_tensor.shape());
    //             // let input = input_batch2.index_select(0, self.index as u64).unwrap();
                
    //             let input_tensor = scaled_input_tensor.get(self.index).unwrap();
              
    //             // println!("input_tensor: {:?}", input_tensor.shape());
    //             input_tensor_o = Some(input_tensor.clone());
    //             // let input_tensor = input_tensor.to_dtype(DType::F32).unwrap();
    //             // value = self.dataset_read(&pv,self.db.clone(),&input_tensor);
    //             // let v = match value{
    //             //     Some(v) => v,
    //             //     None => continue,
    //             // };
    //         } else {
    //         error!("Failed to get next batch from train_batcher");
    //         }
    //         let input_tensor = match input_tensor_o {
    //             Some(t) => t,
    //             None => {
    //                 error!("Failed to get input tensor");
    //                 continue;
    //             },
    //         };
    //         // let len = input_tensor.shape().dims()[0];
    //         // println!("input_tensor len: {:?}", len);
    //         // for it in 0..len {
    //         let valf32:f32 = input_tensor.to_scalar().unwrap();
    //         let val = valf32 as f64;
    //               println!("input_tensor[{}]: {}", pv.key,val);
    //             // let itv = input_tensor.get().unwrap().to_scalar::<f64>().unwrap();
    //             // value = Some(itv);
    //             value = Some(val);
    //             let ret_value = tokio::select! {
    //                 res = self.connection.send(value) => res,
    //                 _ = self.shutdown.recv() => {
    //                     info!("[Publisher]shutdown");
    //                     // If a shutdown signal is received, return from `run`.
    //                     // This will result in the task terminating.
    //                     return Ok(());
    //                 }
    //             };
    //             time::sleep(Duration::from_millis(1)).await;

    //         // }

    //             // let key  = &self.pv.key;
    //             // let value = match self.db.get_front(&key[..]){
    //             //     None => {
    //             //         error!("Publisher][{}] db pop error",key);
    //             //         None
    //             //     },
    //             //     Some(item) => {
    //             //         // println!("[DB_Reader][{}] value: {}/ time: {:?}",key, item.value, item.time_stamp);
    //             //         Some(item.value)
    //             //     },
    //             // };

    //         // time::sleep(Duration::from_millis(1000)).await;
    //         let dt = start.elapsed(); // 시작 시점부터 현재까지의 경과 시간 계산
    //         println!("[Publisher] dt: {:?}", dt);
    //         let dur = Duration::from_millis(1000).saturating_sub(dt); // 1초에서 경과 시간을 뺌
    //         // let dur = Duration::from_millis(200).saturating_sub(dt); // 1초에서 경과 시간을 뺌
    //         // let dur = Duration::from_micros(100).saturating_sub(dt); // 1초에서 경과 시간을 뺌
    //         sleep(dur).await; // 

    //     }
    //     return Ok(())
    // }

}
