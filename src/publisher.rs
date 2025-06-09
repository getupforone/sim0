
use std::collections::HashMap;
use crate::database::{Db,RingBuff,Item};
use crate::process_variable::{PV,PVContext,PVValue};
use tokio::sync::{mpsc,broadcast};
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

// use anyhow::{anyhow, Result};
use candle_core::{Device, Result as cResult, Tensor, D};
use candle_datasets::{batcher::IterResult2, Batcher};

use crate::config::Config;

use tqdm::pbar;

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
        let train_dataset = SCDatasetBuilder::new("train",
             &self.device, data_window,self.config.batch_size)
            .load_data()
            .build();
        let dev = self.device.clone();
        let train_loader = SCDataLoader::new(
                train_dataset.clone(),
               1,
                false,
                true,
                self.config.input_window,
                self.config.output_window,
                &dev,
            );   
        let mut train_batcher = train_loader.batcher();

        // let mut shutdown = Shutdown::new(self.notify_shutdown.subscribe());

        while !self.shutdown.is_shutdown() {
            let start = Instant::now();
            let pv = self.pv.clone();
            let mut value = None;
            let mut input_tensor_o= None;
            if let Some(Ok((input_batch, target_batch))) = train_batcher.next() {
                // println!("input_batch: {:?}, target_batch: {:?}", input_batch.shape(), target_batch.shape());

                let input_batch2 = input_batch.to_device(&self.device).unwrap();
                let target_batch2 = target_batch.to_device(&self.device).unwrap();
                let input_batch2  = input_batch2.squeeze(0).unwrap();
                let input_batch2  = input_batch2.squeeze(1).unwrap();
                // let input = input_batch2.index_select(0, self.index as u64).unwrap();
                let input_tensor = input_batch2.get(self.index).unwrap();
              
                // println!("input_tensor: {:?}", input_tensor.shape());
                input_tensor_o = Some(input_tensor.clone());
                // let input_tensor = input_tensor.to_dtype(DType::F32).unwrap();
                // value = self.dataset_read(&pv,self.db.clone(),&input_tensor);
                // let v = match value{
                //     Some(v) => v,
                //     None => continue,
                // };
            } else {
            error!("Failed to get next batch from train_batcher");
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
                  println!("input_tensor[{}]: {}", pv.key,val);
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
            let dur = Duration::from_millis(200).saturating_sub(dt); // 1초에서 경과 시간을 뺌
            // let dur = Duration::from_micros(100).saturating_sub(dt); // 1초에서 경과 시간을 뺌
            sleep(dur).await; // 

        }
        return Ok(())
    }
}
