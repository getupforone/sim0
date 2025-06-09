use crate::database::{Db,RingBuff,Item};
use crate::process_variable::{PVContext,PVValue};
use tokio::sync::{mpsc,broadcast};
use std::sync::{Arc, Mutex};
use tokio::task::JoinError;
use tracing::{debug, error, info, instrument};
use crate::shutdown::Shutdown;
use tokio::time::Instant;
use tokio::time::{sleep,Duration};
use anyhow::{anyhow, Result};

// pub type Error = Box<dyn std::error::Error + Send + Sync>;
// pub type Result<T> = std::result::Result<T, Error>;
const READ_BIN_CODE: u8 = 0b01;

use crate::process_variable::PV;
use std::collections::HashMap;

#[derive(Debug)]
pub struct PV_RX {
    pvdata_rx: mpsc::Receiver<PVValue>,
}
impl PV_RX{
    async fn get_pv(&mut self) -> Option<PVValue> 
    {
        // println!("[PV_RX]get_pv");
        let pvvalue_o = self.pvdata_rx.recv().await;
        return pvvalue_o;
    }
}

#[derive(Debug)]
pub struct Listener {
    db: Db,
    // notify_shutdown: broadcast::Sender<()>,
    shutdown: Shutdown,
    shutdown_complete_tx: mpsc::Sender<()>,
    
    buffer_len : usize,
    buffer_duration :u64,
    // pvdata_rx: mpsc::Receiver<PVValue>,
    connection: PV_RX,
    pv_map: Arc<Mutex<HashMap<String, PV>>>,
}
impl Listener{
    pub fn new(db: Db,shutdown: Shutdown,shutdown_complete_tx: mpsc::Sender<()>,
     buffer_len:usize,buffer_duration:u64,mut pvdata_rx: mpsc::Receiver<PVValue>,pvm:&HashMap<String,PV>)->Listener{
        let listener = Listener{
            db:db,
            shutdown,
            shutdown_complete_tx,
            buffer_len,
            buffer_duration,
            // pvdata_rx,
            connection: PV_RX {pvdata_rx},
            pv_map: Arc::new(Mutex::new(pvm.clone())),

        };
        // let pvc = pvcontext.clone();
        // if !listener.db.contained(&pvc.key) {
        //     println!("[Listener][new] [set] key = {:?}", pvc.key.clone());
        //     listener.db.set(pvc.key,RingBuff::new(buffer_len));
        // }
        listener

    }
    pub fn db_set_push(&self, pvvalue_o: &Option<PVValue>)-> Result<f64>{
        
        // let pvvalue_o = self.pvdata_rx.recv().await;
        // let pvvalue_o = self.connection.get_pv().await;
       
        let pvvalue = match pvvalue_o {
            Some(res) => res,
            None => {
                // println!("[Listener]recv_db_set_push error");
                return Err(anyhow!("[Listener]recv_db_set_push error"));
                // return Ok(0.0);
            },
        };
        // let pvvalue = pvvalue_o.unwrap();
        // let key = pvvalue.key.clone();
        let key =&pvvalue.key;
        
        if !self.db.contained(key){
                info!("[Listener][new] [set] key = {:?}", key);
                self.db.set(key,RingBuff::new(pvvalue.buff_size));
        }
      
        // let key  = pvvalue.key.clone();
        let value = pvvalue.value.unwrap();
        if let Some(timestamp) = pvvalue.time_stamp{
            
            self.db.push(&key[..], Item{value:value, time_stamp:timestamp});
            // info!("[Listener][{:?}]: key: {}/ value: {}",Instant::now(), key, value);
        }
        Ok(value)
    }

    pub async fn run(&mut self) -> Result<()> {

        // let mut shutdown = Shutdown::new(self.notify_shutdown.subscribe());
        // db.set(String::from("pv1"),RingBuff::new(3),Some(Duration::from_millis(1000)));
        while !self.shutdown.is_shutdown() {
            let start = Instant::now();

            let pvvalue_o: Option<PVValue> = tokio::select!{
                res = self.connection.get_pv() => res,

                _ = self.shutdown.recv() => {
                    info!("[Listener]shutdown");
                    // If a shutdown signal is received, return from `run`.
                    // This will result in the task terminating.
                    return Ok(());
                }
            };
            let pvvalue = match pvvalue_o.clone() {
                Some(res) => res,
                None => {
                    // println!("[Listener]recv error");
                    return Err(anyhow!("[Listener]recv error"));
                },
            };
            // // pvvalue.
            // let pv_map_guard = self.pv_map.lock().unwrap();
            // let pv = pv_map_guard.get(&pvvalue.key)
            //     .ok_or_else(|| anyhow!("PV not found in map: {}", pvvalue.pvname))?;
            if pvvalue.enable_ca == false{continue;}
            
            if pvvalue.wr  &  READ_BIN_CODE  == 0 {println!("[Listener]recv pvvalue: {:?}", pvvalue.key);continue;}
            // info!("[Listener]pvvalue_o: {:?}", pvvalue_o);
            match self.db_set_push(&pvvalue_o){
                Ok( value) => {
                    // info!("value put to db in listner server: {}", value);
                },
                Err(err) =>{error!("Error in listner server: {}", err);}
            }

            let dt = start.elapsed(); // 시작 시점부터 현재까지의 경과 시간 계산
            // println!("[inferer] dt: {:?}", dt);
            let dur = Duration::from_millis(40).saturating_sub(dt); // 1초에서 경과 시간을 뺌
            sleep(dur).await; // 
        }
        
        return Ok(())
    }
}

#[derive(Debug)]
struct Handler {
    db: Db,
    pvvalue: PVValue,
    shutdown: Shutdown,
    _shutdown_complete: mpsc::Sender<()>,
}
