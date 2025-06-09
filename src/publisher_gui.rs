
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
use crate::database::KVT;

#[derive(Debug)]
pub struct OUT_TX {
    pv:PV,
    eng_tx: mpsc::Sender<KVT>,
}
impl OUT_TX{
    pub async fn send(&mut self,maybe_value: Option<f64>)-> std::result::Result<(),PublisherGUIError>
    {
        let eng_tx = self.eng_tx.clone();
        let pv = self.pv.clone();
        let ret = match maybe_value{
            Some(value)=>{
                // println!("[OUT_TX][{}] value: {}", pv.key, value);
                let handle = tokio::spawn(async move{

                    let kvt = KVT{
                        key: pv.key,
                        value: value,
                        time_stamp:Utc::now(),
                    };
                    eng_tx.send(kvt).await
                });
                let ret_val= handle.await?;
                ret_val?
            }
            None => {

                error!("dbpop is failed. so skip the sending");
            }
        };
        Ok(ret)
    }
}

use std::fmt;
#[derive(Debug)]
pub enum PublisherGUIError{
    SendError,
    JoinError,
}

impl std::error::Error for PublisherGUIError {}

impl fmt::Display for PublisherGUIError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result{
        match self {
            PublisherGUIError::SendError => write!(f, "Send Error"),
            PublisherGUIError::JoinError => write!(f, "Joint Error"),
        }
    }
}
impl From<mpsc::error::SendError<KVT>> for PublisherGUIError{
    fn from(_: mpsc::error::SendError<KVT>) -> Self{
        PublisherGUIError::SendError
    } 
}
impl From<JoinError> for PublisherGUIError{
    fn from(_: JoinError) -> Self{
        PublisherGUIError::JoinError
    } 
}

#[derive(Debug)]
pub struct PublisherGUI {
     db: Db,
    // notify_shutdown: broadcast::Sender<()>,
    shutdown: Shutdown,
    shutdown_complete_tx: mpsc::Sender<()>,
    connection: OUT_TX,
    pv:PV,
}
impl PublisherGUI {
    pub fn new(db: Db, shutdown: Shutdown, shutdown_complete_tx: mpsc::Sender<()>,
         pv:&PV,
         eng_tx: mpsc::Sender<KVT>
        )->PublisherGUI
    {
        PublisherGUI{
            db,
            // notify_shutdown,
            shutdown,
            shutdown_complete_tx,
            connection: OUT_TX{pv:pv.clone(), eng_tx},
            pv:pv.clone(),
        }
    }
    pub  fn db_read(&mut self,pv:&PV,db:Db)-> Option<f64>{
        let key  = pv.key.clone();
        // let value = match db.get_front(&key[..]){
        let value = match db.pop(&key[..]){
            None => {
                error!("PublisherGUI][{}] db pop error",key);
                None
            },
            Some(item) => {
                // println!("[DB_Reader][{}] value: {}/ time: {:?}",key, item.value, item.time_stamp);
                Some(item.value)
            },
        };
        value
    }
    
    pub async fn run(&mut self) -> Result<()> {

        // let mut shutdown = Shutdown::new(self.notify_shutdown.subscribe());

        while !self.shutdown.is_shutdown() {
            let start = Instant::now();

            let pv = self.pv.clone();
            let value = self.db_read(&pv,self.db.clone());
                // let key  = &self.pv.key;
                // let value = match self.db.get_front(&key[..]){
            let v = match value{
                Some(v) => v,
                None => continue,
            };
                //     None => {
                //         error!("PublisherGUI][{}] db pop error",key);
                //         None
                //     },
                //     Some(item) => {
                //         // println!("[DB_Reader][{}] value: {}/ time: {:?}",key, item.value, item.time_stamp);
                //         Some(item.value)
                //     },
                // };
            let ret_value = tokio::select! {
                res = self.connection.send(value) => res,
                _ = self.shutdown.recv() => {
                    info!("[PublisherGUI]shutdown");
                    // If a shutdown signal is received, return from `run`.
                    // This will result in the task terminating.
                    return Ok(());
                }
            };
            // time::sleep(Duration::from_millis(1000)).await;
            let dt = start.elapsed(); // 시작 시점부터 현재까지의 경과 시간 계산
            println!("[PublisherGUI] dt: {:?}", dt);
            let dur = Duration::from_millis(100).saturating_sub(dt); // 1초에서 경과 시간을 뺌
            // let dur = Duration::from_millis(40).saturating_sub(dt); // 1초에서 경과 시간을 뺌
            // let dur = Duration::from_micros(100).saturating_sub(dt); // 1초에서 경과 시간을 뺌
            sleep(dur).await; // 

        }
        return Ok(())
    }
}
