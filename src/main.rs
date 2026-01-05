use std::collections::HashMap;
use std::future::Future;
use std::sync::Arc;
use iced::futures::channel::mpsc::Receiver;
use plotters::chart;
use tokio::signal;
use tokio::time::Instant;
use tokio::sync::{broadcast,mpsc, watch};
use tokio::net::UdpSocket;
use tracing::{debug, error, info, instrument};
use tokio::time::{self, Duration};


use tokio::time::{sleep};

use tokio::runtime::Builder;


// extern crate iced;
// extern crate plotters;
// extern crate sysinfo;

use chrono::{DateTime, Timelike, Utc, Duration as cDuration};
use iced::{
    alignment::{Horizontal, Vertical},
    font,
    widget::{
        canvas::{Cache, Frame, Geometry},
        Column, Container, Row, Scrollable, Space, Text,
    },
    Alignment, Element, Font, Length, Size, Task,
};
use plotters::prelude::ChartBuilder;
use plotters_backend::DrawingBackend;
use plotters_iced::{Chart, ChartWidget, Renderer};


use std::{
    collections::VecDeque,
    // time::{Duration, Instant},
};
use sysinfo::{CpuRefreshKind, RefreshKind, System};

use std::fs;

use anyhow::{anyhow, Result};

use candle_core::{DType,Device, Tensor, D};
// pub type Error = Box<dyn std::error::Error + Send + Sync>;
// pub type Result<T> = std::result::Result<T, Error>;

pub mod database;
use database::DbDropGuard;
use database::Db;

use database::Item;
use database::KVT;
use crate::database::RingBuff;
// 
pub mod process_variable;
use process_variable::PV;
use process_variable::PVMap;
use process_variable::{PVContext,PVContextVec};
use process_variable::PVInfos;

use crate::process_variable::{PVValue,PVVec};

pub mod shutdown;
use shutdown::Shutdown;

pub mod listner;
use listner::Listener;

pub mod publisher;
pub mod publisher_gui;
use publisher::Publisher;
use publisher_gui::PublisherGUI;

pub mod dataset;
use dataset::{SCDataLoader,SCDatasetBuilder};
// use dataset::{SCShotDataLoader};
pub mod config;
use config::Config;

pub mod savedata;


// fn main() {
//     println!("Hello, world!");
// }

#[derive(Debug)]
struct Server{
    db_holder: DbDropGuard,
    pvs:PVMap,
    notify_shutdown: broadcast::Sender<()>,
    shutdown_complete_tx: mpsc::Sender<()>,
    eng_tx:mpsc::Sender<KVT>,
    gui_rx:mpsc::Receiver<KVT>,
    device: Device,
    config: Config,

}
const READ_BIN_CODE: u8 = 0b01;
const WRITE_BIN_CODE: u8 = 0b10;
const BUFF_LEN: usize = 10;
const BUFF_DURATION:u64 = 10000;
// async fn pv_put(mut outdata_rx: mpsc::Receiver<PVValue>) -> Result<()>


async fn pv_process( outdata_rx: &mut mpsc::Receiver<PVValue>, 
    pvs:  &HashMap<String, PV>,
    pvdata_tx:& mpsc::Sender<PVValue>,
    eng_tx:&mpsc::Sender<KVT>,
    gui_rx:&mut mpsc::Receiver<KVT>,)-> Result<()>
{

    // if let Some(value) = outdata_rx.recv().await{

    //     let pvvalue = value;
    //     let pv = PV{key:pvvalue.key.clone(),pvname: pvvalue.pvname.clone(),wr:0b11,enable_ca:pvvalue.enable_ca,buff_size:pvvalue.buff_size,vec_size:pvvalue.vec_size,pred_size:pvvalue.pred_size};
    //     let pvctx = PVContext::new(pv.clone()).await;
    //     let channel =  Arc::clone(&pvctx.channel);
    //     let handle = pvctx.clone().put(pvvalue.value.unwrap(), &*channel ).await?;
    //     println!("[pv_put]key = {}/pvname = {}",pv.key, pv.pvname);
    // }
    // let s1 = Instant::now();
    // pvs_get(pvs, pvdata_tx,eng_tx).await?;
    // let dt1 = s1.elapsed(); 
    // let s2 = Instant::now();
    pvs_put(outdata_rx,pvs).await?;
    // let dt2 = s2.elapsed(); 
    // println!("[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[main] dt1: {:?}", dt1);
    // println!("[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[main] dt2: {:?}", dt2);
    // pv_put(outdata_rx).await?;
    Ok(())
}
// async fn pv_put( outdata_rx: &mut mpsc::Receiver<PVValue>) -> Result<()>
// {
//     let pvvalue_o = outdata_rx.recv().await;
//     let pvvalue = match pvvalue_o {
//         Some(value) => value,
//         None => return Ok(()),
//     };
//     let pv = PV{key:pvvalue.key.clone(),pvname: pvvalue.pvname.clone(),wr:0b11,enable_ca:pvvalue.enable_ca,buff_size:pvvalue.buff_size,vec_size:pvvalue.vec_size,pred_size:pvvalue.pred_size};
//     if pv.enable_ca == false{
//         return Ok(());
//     }
//     if pv.wr  &  WRITE_BIN_CODE  == 0 {
//         return Ok(());
//     }
//     let pvctx = PVContext::new(pv.clone()).await;
//     let channel =  Arc::clone(&pvctx.channel);
//     let handle = pvctx.clone().put(pvvalue.value.unwrap(), &*channel ).await?;
//     // println!("[pv_put]key = {}/pvname = {}",pv.key, pv.pvname);
//     // let ret_val = handle.await?;
//     Ok(())
// }


// async fn pvs_put( outdata_rx: &mut mpsc::Receiver<PVValue>,eng_tx:&mpsc::Sender<KVT>) -> Result<()>
async fn pvs_put( outdata_rx: &mut mpsc::Receiver<PVValue>,
        pvs:  &HashMap<String, PV>,) -> Result<()>
{
    let limit = 50;
    let mut buffer: Vec<PVValue> = Vec::with_capacity(limit);
    let n = outdata_rx.recv_many(&mut buffer, limit).await;
    
    let pvvalue_o = outdata_rx.recv().await;
    let pvvalue = match pvvalue_o {
        Some(value) => value,
        None => return Ok(()),
    };
    for i in 0..n{
        let pvvalue = buffer[i].clone();

        let pv = PV{key:pvvalue.key.clone(),
            pvname: pvvalue.pvname.clone(),
            wr:pvvalue.wr,
            enable_ca:pvvalue.enable_ca,
            buff_size:pvvalue.buff_size,
            vec_size:pvvalue.vec_size,
            pred_size:pvvalue.pred_size};
        
        // if pv.enable_ca == false{continue;}
        // if pv.wr  &  WRITE_BIN_CODE  == 0 {continue;}
        if pv.enable_ca == false{return Ok(());}
        // if pv.wr  &  WRITE_BIN_CODE  == 0 {return Ok(());}
        
        let mut pvctx = PVContext::new(pv).await;
        let channel =  Arc::clone(&pvctx.channel);
        // let gui_tx_val = KVT{key:pvctx.key,value:pvvalue.value.unwrap(),time_stamp:Utc::now()};
        let handle = pvctx.put(pvvalue.value.unwrap(), &*channel ).await?;


    }

    
    Ok(())
}
async fn pvput_startstop( startstop:bool) -> Result<()>
{

    let pv = PV{key:"SIM0_STARTSTOP".to_string(),
        pvname: "SIM0_STARTSTOP".to_string(),
        wr:0b11,
        enable_ca:true,
        buff_size:1,
        vec_size:1,
        pred_size:1,
    };
    
    // if pv.enable_ca == false{continue;}
    // if pv.wr  &  WRITE_BIN_CODE  == 0 {continue;}
    if pv.enable_ca == false{return Ok(());}
    // if pv.wr  &  WRITE_BIN_CODE  == 0 {return Ok(());}
    
    let mut pvctx = PVContext::new(pv).await;
    let channel =  Arc::clone(&pvctx.channel);
    // let gui_tx_val = KVT{key:pvctx.key,value:pvvalue.value.unwrap(),time_stamp:Utc::now()};
    let val = if startstop == true{ 1.0}else{0.0};
    let handle = pvctx.put(val, &*channel ).await?;

    Ok(())
}

impl Server{
    pub fn new(pv_infos:HashMap<String, PVInfos>,
        notify_shutdown: broadcast::Sender<()>,
        shutdown_complete_tx: mpsc::Sender<()>,
        eng_tx:mpsc::Sender<KVT>,
        gui_rx:mpsc::Receiver<KVT>,
        device:Device,
        config:Config,
        )->Self{
            let mut server = Server{
                db_holder:DbDropGuard::new(),
                pvs:PVMap::new(pv_infos),
                notify_shutdown,
                shutdown_complete_tx,
                eng_tx,
                gui_rx,
                device,
                config,
            };
            
            server
    }   
    
    
    async fn run(&mut self)->Result<()>{
        let mut shutdown = Shutdown::new(self.notify_shutdown.subscribe());

        for (k,pv) in self.pvs.pv_map.iter()
        {
            // let key = k.clone();
            if !self.db_holder.db().contained(&k){
                info!("[Server][run] [set] key = {:?}", k);
                self.db_holder.db().set(k,RingBuff::new(pv.buff_size));

                for idx in 0..pv.pred_size{
                    let zero_item = Item{
                        value:0.0_f64,
                        time_stamp: Utc::now(),
                    };
                    self.db_holder.db().push(&k[..],zero_item);
                }
            }
        }

        let (pvdata_tx, mut pvdata_rx) = mpsc::channel(100);
        let (outdata_tx, mut outdata_rx) = mpsc::channel(100);
        let (startstop_tx, mut startstop_rx) = watch::channel(false);
       

        // let mut listener: Listener = Listener::new(
        //     self.db_holder.db(),Shutdown::new(self.notify_shutdown.subscribe()),
        //     self.shutdown_complete_tx.clone(),
        //     BUFF_LEN, BUFF_DURATION,pvdata_rx,&self.pvs.pv_map);

        // tokio::spawn(async move{
        //     if let Err(err) = listener.run().await{
        //         eprintln!("Error in run_listener: {}", err);
        //         error!("Error in run server: {}", err);
        //     }
        // });




        let data_window = self.config.input_window + self.config.output_window;


        // if let Some(Ok((input_batch, target_batch))) = test_batcher.next() {
        //     println!("input_batch: {:?}, target_batch: {:?}", input_batch.shape(), target_batch.shape());

            // let input_batch2 = input_batch.to_device(&self.device).unwrap();
            // let target_batch2 = target_batch.to_device(&self.device).unwrap();
            
            // println!("input: {:?}, target: {:?}", input, target);
            // Do something with the input and target tensors
       
            for (key,pv) in self.pvs.pv_map.iter(){
                let o_tx = outdata_tx.clone();
                

                let conf = self.config.clone();
                let dev  = self.device.clone();
                let mut publisher = Publisher::new(
                    self.db_holder.db(),
                    Shutdown::new(self.notify_shutdown.subscribe()),
                    self.shutdown_complete_tx.clone(),pv,o_tx,
                    startstop_tx.clone(),
                    conf,dev,
                    );
                tokio::spawn(async move{
                    if let Err(err) = publisher.run().await{
                        eprintln!("Error in run talker: {}", err);
                        error!("Error in run server: {}", err);
                    }
                });

                // let etx =  self.eng_tx.clone();
                // let mut publisher_gui = PublisherGUI::new(
                //     self.db_holder.db(),
                //     Shutdown::new(self.notify_shutdown.subscribe()),
                //     self.shutdown_complete_tx.clone(),pv,etx);
                // tokio::spawn(async move{
                //     if let Err(err) = publisher_gui.run().await{
                //         eprintln!("Error in run talker: {}", err);
                //         error!("Error in run server: {}", err);
                //     }
                // });



            }
        //  } else {
        //     error!("Failed to get next batch from train_batcher");
        // }
        // let pvm: HashMap<String, PV> = self.pvs.pv_map.clone();
        let socket = UdpSocket::bind("0.0.0.0:0").await?;
        let server = "127.0.0.1:9000";
        tokio::spawn(async move{
            loop{
                // let start = Instant::now();
                let mut val = false;
                let startstop = startstop_rx.changed().await;
                val = match startstop {
                    Ok(_) => {
                        let val = *startstop_rx.borrow();
                        info!("startstop changed: {}", val);
                        val
                    },
                    Err(e) => {
                        error!("startstop channel closed: {}", e);
                        break;
                    }
                };
                if val {
                            socket.send_to(b"START", server).await;
                        } else {
                            socket.send_to(b"STOP", server).await;
                        }
                // let dt = start.elapsed(); // 시작 시점부터 현재까지의 경과 시간 계산
                // println!("[startstopwatch] dt: {:?}", dt);
                // let dur = Duration::from_millis(1).saturating_sub(dt); // 1초에서 경과 시간을 뺌
                // // let dur = Duration::from_millis(200).saturating_sub(dt); // 1초에서 경과 시간을 뺌
                // // let dur = Duration::from_micros(100).saturating_sub(dt); // 1초에서 경과 시간을 뺌
                // sleep(dur).await; // 
            }
        });

        let pvm = &self.pvs.pv_map;


        
        // let pvvvec1 = pvm.get("pvvec1").cloned();
        // let mut shutdown2 = Shutdown::new(self.notify_shutdown.subscribe());
        while !shutdown.is_shutdown() {
            // let start = Instant::now();
            let ptx = pvdata_tx.clone();
            let etx =  self.eng_tx.clone();
            tokio::select!{
                res = pv_process(&mut outdata_rx, pvm, &ptx, &etx,&mut self.gui_rx) =>{
                    if let Err(err) = res{
                        error!("Error in pv_process: {}", err);

                    }
                }
                // () = &mut sleep => {
                //     println!("timer elapsed");
                //     sleep.as_mut().reset(Instant::now() + Duration::from_millis(500));
                // },
                _ = shutdown.recv() => {
                    info!("shutdown ");
                    // If a shutdown signal is received, return from `run`.
                    // This will result in the task terminating.
                    return Ok(());
                }
            }




            // let dt = start.elapsed(); // 시작 시점부터 현재까지의 경과 시간 계산
            // println!("====================[main]==================== dt: {:?}", dt);
            // let dur = Duration::from_millis(100).saturating_sub(dt); // 1초에서 경과 시간을 뺌
            // let dur = Duration::from_micros(100).saturating_sub(dt); // 1초에서 경과 시간을 뺌
            // sleep(dur).await; // 
    }


    return Ok(())
}

}

pub async fn run(pv_infos: HashMap<String, PVInfos>,eng_tx:mpsc::Sender<KVT>,
    gui_rx:mpsc::Receiver<KVT>,config:Config,device:Device, shutdown: impl Future)
{
    println!("test1");
    for (pv_key, pv_info) in pv_infos.iter(){
        println!("key = {}/pvname = {}",pv_key, pv_info.pvname);
    }
    println!("test2");
    let (notify_shutdown,_)  = broadcast::channel(1);
    let (shutdown_complete_tx, mut shutdown_complete_rx) = mpsc::channel(1);

    let mut server = Server::new(pv_infos.clone(),notify_shutdown, shutdown_complete_tx,eng_tx,gui_rx,device,config);

    tokio::select!{
        res = server.run() => {
            if let Err(err) = res{
                println!("fail server");
                error!("Error in run server: {}", err);
            }
        }
        _ = shutdown => {
            info!("[run]shutdown signal accepted ");
            println!("shuttting down");
        }
        
    }
    let Server { shutdown_complete_tx ,notify_shutdown, ..} = server;
    // when all Sender handles have been dropped, no new values may be sent. At this point, the channel is "closed".
    //Once a receiver has received all values retained by the channel, the next call to recv will return with RecvError::Closed.
    drop(notify_shutdown);
    // when all Sender handles have been dropped, it is no longer possible to send values into the channel. This is considered 
    //the termination  event of the stream. As such , Receiver::poll return Ok(Ready(None))
    drop(shutdown_complete_tx);
    let _ = shutdown_complete_rx.recv().await;

}

pub fn main() -> Result<()>
{    
    tracing_subscriber::fmt::init();
    println!("=================");
    println!("|     sim0       |");
    println!("=================");
    let model_name = "rec_trfm";
    println!("model name   = {}",model_name);


    // let tril = Tensor::tril2(5, DType::F32, &Device::Cpu)?;
    // println!("tril {}", tril);
    // let device: Device = Device::Cpu;
    let device: Device = Device::cuda_if_available(0)?;
    

    let mut cfg = match model_name {
        "trfm" => {
            Config::trfm()
        },
        "rec_trfm" =>{
            Config::rectrfm()
        },
        &_ => todo!()
    };
     let data_window = cfg.input_window + cfg.output_window;

    let pvmap:HashMap<String, PVInfos> =  {
        let fdata: String = fs::read_to_string("./pv_map.json").expect("error reading file");
        serde_json::from_str(&fdata).unwrap()
    };

    let (eng_tx, eng_rx) = mpsc::channel(16);
    let (gui_tx, gui_rx) = mpsc::channel(16);

    // println!("{:?}",pvmap);
    let rt = Builder::new_multi_thread()
    .enable_all()
    .build()
    .unwrap();
    // run(pvmap, signal::ctrl_c()).await;
  
    // std::thread::spawn(move || {
        rt.block_on(async move {
            run(pvmap,eng_tx,gui_rx, cfg, device,signal::ctrl_c()).await;
        });
    // });

    //     iced::application("A cool app", MyApp::update, MyApp::view)
    // .theme(MyApp::theme)
    // .run();

    // let state_new_closure = move || {State::new(gui_tx,eng_rx)};
    // iced::application("Sim0", State::update, State::view)
    // .antialiasing(true)
    // .default_font(Font::with_name("Noto Sans"))
    // .subscription(|_| {
    //     const FPS: u64 = 10;
    //     iced::time::every(Duration::from_millis(1000 / FPS)).map(|_| Message::Tick)
    // })
    // .window(iced::window::Settings { size: iced::Size{width:600f32,height:600f32}, ..Default::default() }
        
    // )
    // .run_with(state_new_closure)
    // .unwrap();

    Ok(())

}


// pub fn main() -> Result<()>
// {    
//     tracing_subscriber::fmt::init();
//     println!("=================");
//     println!("|     sim0       |");
//     println!("=================");
//     let model_name = "rec_trfm";
//     println!("model name   = {}",model_name);


//     // let tril = Tensor::tril2(5, DType::F32, &Device::Cpu)?;
//     // println!("tril {}", tril);
//     // let device: Device = Device::Cpu;
//     let device: Device = Device::cuda_if_available(0)?;
    

//     let mut cfg = match model_name {
//         "trfm" => {
//             Config::trfm()
//         },
//         "rec_trfm" =>{
//             Config::rectrfm()
//         },
//         &_ => todo!()
//     };
//      let data_window = cfg.input_window + cfg.output_window;

//     let pvmap:HashMap<String, PVInfos> =  {
//         let fdata: String = fs::read_to_string("./pv_map.json").expect("error reading file");
//         serde_json::from_str(&fdata).unwrap()
//     };

//     let (eng_tx, eng_rx) = mpsc::channel(16);
//     let (gui_tx, gui_rx) = mpsc::channel(16);

//     // println!("{:?}",pvmap);
//     let rt = Builder::new_multi_thread()
//     .enable_all()
//     .build()
//     .unwrap();
//     // run(pvmap, signal::ctrl_c()).await;
  
//     std::thread::spawn(move || {
//         rt.block_on(async move {
//             run(pvmap,eng_tx,gui_rx, cfg, device,signal::ctrl_c()).await;
//         });
//     });

//     //     iced::application("A cool app", MyApp::update, MyApp::view)
//     // .theme(MyApp::theme)
//     // .run();

//     // let state_new_closure = move || {State::new(gui_tx,eng_rx)};
//     // iced::application("Sim0", State::update, State::view)
//     // .antialiasing(true)
//     // .default_font(Font::with_name("Noto Sans"))
//     // .subscription(|_| {
//     //     const FPS: u64 = 10;
//     //     iced::time::every(Duration::from_millis(1000 / FPS)).map(|_| Message::Tick)
//     // })
//     // .window(iced::window::Settings { size: iced::Size{width:600f32,height:600f32}, ..Default::default() }
        
//     // )
//     // .run_with(state_new_closure)
//     // .unwrap();

//     Ok(())

// }

// const PLOT_SECONDS: usize = 60; //1 min
// const TITLE_FONT_SIZE: u16 = 22;
// const SAMPLE_EVERY: Duration = Duration::from_millis(100);
// // 
// const FONT_BOLD: Font = Font {
//     family: font::Family::Name("Noto Sans"),
//     weight: font::Weight::Bold,
//     ..Font::DEFAULT
// };


// #[derive(Debug)]
// enum Message {
//     /// message that cause charts' data lazily updated
//     Tick,
//     FontLoaded(std::result::Result<(), font::Error>),
// }

// struct State {
//     chart: SystemChart,
//     gui_tx: mpsc::Sender<KVT>, 
//     eng_rx: mpsc::Receiver<KVT>, 
    
// }

// impl State {
//     fn new(gui_tx:mpsc::Sender<KVT>,eng_rx:mpsc::Receiver<KVT>) -> (Self, Task<Message>) {
//         (
//             Self {
//                 chart: Default::default(),
//                 gui_tx,
//                 eng_rx,

//             },
//             Task::batch([
//                 font::load(include_bytes!("./fonts/notosans-regular.ttf").as_slice())
//                     .map(Message::FontLoaded),
//                 font::load(include_bytes!("./fonts/notosans-bold.ttf").as_slice())
//                     .map(Message::FontLoaded),
//             ]),
            
//         )
//     }

//     fn update(&mut self, message: Message) {
//         match message {
//             Message::Tick => {
//                 self.chart.update(&mut self.eng_rx);
//             }
//             _ => {}
//         }
//     }

//     fn view(&self) -> Element<'_, Message> {
//         let content = Column::new()
//             .spacing(20)
//             .align_x(Alignment::Center)
//             .width(Length::Fill)
//             .height(Length::Fill)
//             .push(
//                 Text::new("Data Graph")
//                     .size(TITLE_FONT_SIZE)
//                     .font(FONT_BOLD),
//             )
//             .push(self.chart.view());

//         Container::new(content)
//             //.style(style::Container)
//             .padding(5)
//             .center_x(Length::Fill)
//             .center_y(Length::Fill)
//             .into()
//     }
// }

// struct SystemChart {
//     sys: System,
//     last_sample_time: Instant,
//     items_per_row: usize,
//     processors: HashMap<String,CpuUsageChart>,
//     chart_height: f32,
//     // pvmap:HashMap<String, PVInfos>,
//     rt: tokio::runtime::Runtime,
// }

// impl Default for SystemChart {
//     fn default() -> Self {
//         Self {
//             sys: System::new_with_specifics(
//                 RefreshKind::new().with_cpu(CpuRefreshKind::new().with_cpu_usage()),
//             ),
//             last_sample_time: Instant::now(),
//             items_per_row: 1,
//             processors: Default::default(),
//             chart_height: 500.0,
//             // pvmap :  {
//             //     let fdata: String = fs::read_to_string("./pv_map.json").expect("error reading file");
//             //     serde_json::from_str(&fdata).unwrap()
//             // },
//             rt : tokio::runtime::Builder::new_current_thread()
//             .enable_all()
//             .build().unwrap(),
//         }
//     }
// }

// impl SystemChart {
//     #[inline]
//     fn is_initialized(&self) -> bool {
//         !self.processors.is_empty()
//     }

//     #[inline]
//     fn should_update(&self) -> bool {
//         !self.is_initialized() || self.last_sample_time.elapsed() >= SAMPLE_EVERY
//         // !self.is_initialized() 
//     }
//     fn recv_from_eng(&mut self,eng_rx:&mut mpsc::Receiver<KVT>) -> Vec<KVT> {

//         let limit = 2;
//         let mut buffer: Vec<KVT> = Vec::with_capacity(limit);
//         // let n = outdata_rx.recv_many(&mut buffer, limit).await;
//         let mut kvt = None;
//         self.rt.block_on(async {
//             // let n = eng_rx.recv_many(&mut buffer, limit).await;
//             kvt = eng_rx.recv().await;
//         });
//         // buffer
//         let kvt = kvt.unwrap();
//         // println!("[gui]key = {},value = {}", kvt.key,kvt.value);
        
//         buffer.push(kvt);
//         buffer
//     }
//     fn update(&mut self,eng_rx:&mut mpsc::Receiver<KVT>,) {
//         if !self.should_update() {
//             return;
//         }
//         //eprintln!("refresh...");

//         // self.sys.refresh_cpu();
//         self.last_sample_time = Instant::now();
//         let now = Utc::now();
//         let kvt_vec = self.recv_from_eng(eng_rx);

//         // let data = self.sys.cpus().iter().map(|v| v.cpu_usage() as i32);
//         for  kvt in kvt_vec{
//             self.processors.entry(kvt.key).or_insert(
//                 // CpuUsageChart::new(vec![(now, kvt.value)].into_iter()))
//                 CpuUsageChart::new(vec![(kvt.time_stamp, kvt.value)].into_iter()))
//                 .push_data(kvt.time_stamp, kvt.value); 
//         }
//         //check if initialized
//         // if !self.is_initialized() {
//         //     eprintln!("init...");
//         //     let mut processors: Vec<_> = kvt_vec.iter()
//         //         .map(|kvt| CpuUsageChart::new(vec![(now, kvt.value)].into_iter()))
//         //         .collect();
//         //     self.processors.append(&mut processors);

                
//         // } else {
//         //     eprintln!("update...");
//         //     for (kvt, p) in data.zip(self.processors.iter_mut()) {
//         //         p.push_data(now, kvt.value);
//         //     }
//         // }
//     }

//     fn view(&self) -> Element<Message> {
//         if !self.is_initialized() {
//             Text::new("Loading...")
//                 .align_x(Horizontal::Center)
//                 .align_y(Vertical::Center)
//                 .into()
//         } else {
//             let mut col = Column::new()
//                 .width(Length::Fill)
//                 .height(Length::Shrink)
//                 .align_x(Alignment::Center);

//             let chart_height = self.chart_height;
//             let mut idx = 0;
//             let keys:Vec<String> = self.processors.keys().cloned().collect();
            
//             for chunk in keys.chunks(self.items_per_row)  {
//                 let mut row = Row::new()
//                     .spacing(10)
//                     .padding(20)
//                     .width(Length::Fill)
//                     .height(Length::Shrink)
//                     .align_y(Alignment::Center);
//                 for key in chunk {
//                     let item = self.processors.get(key).unwrap();
//                     row = row.push(item.view(key.clone(), chart_height));
//                     idx += 1;
//                 }
//                 while idx % self.items_per_row != 0 {
//                     row = row.push(Space::new(Length::Fill, Length::Fixed(50.0)));
//                     idx += 1;
//                 }
//                 col = col.push(row);
//             }

//             Scrollable::new(col).height(Length::Shrink).into()
//         }
//     }
// }

// struct CpuUsageChart {
//     cache: Cache,
//     data_points: VecDeque<(DateTime<Utc>, f64)>,
//     limit: Duration,
// }

// impl CpuUsageChart {
//     fn new(data: impl Iterator<Item = (DateTime<Utc>, f64)>) -> Self {
//         let data_points: VecDeque<_> = data.collect();
//         Self {
//             cache: Cache::new(),
//             data_points,
//             limit: Duration::from_secs(PLOT_SECONDS as u64),
//         }
//     }

//     fn push_data(&mut self, time: DateTime<Utc>, value: f64) {
//         let cur_ms = time.timestamp_millis();
//         self.data_points.push_front((time, value));
//         loop {
//             if let Some((time, _)) = self.data_points.back() {
//                 let diff = Duration::from_millis((cur_ms - time.timestamp_millis()) as u64);
//                 if diff > self.limit {
//                     self.data_points.pop_back();
//                     continue;
//                 }
//             }
//             break;
//         }
//         self.cache.clear();
//     }

//     fn view(&self, key: String, chart_height: f32) -> Element<Message> {
//         Column::new()
//             .width(Length::Fill)
//             .height(Length::Shrink)
//             .spacing(5)
//             .align_x(Alignment::Center)
//             .push(Text::new(format!("PV {}", key)))
//             .push(ChartWidget::new(self).height(Length::Fixed(chart_height)).width(Length::Fixed(chart_height*2f32)))
//             .into()
//     }
// }

// impl Chart<Message> for CpuUsageChart {
//     type State = ();
//     // fn update(
//     //     &mut self,
//     //     event: Event,
//     //     bounds: Rectangle,
//     //     cursor: Cursor,
//     // ) -> (event::Status, Option<Message>) {
//     //     self.cache.clear();
//     //     (event::Status::Ignored, None)
//     // }

//     #[inline]
//     fn draw<R: Renderer, F: Fn(&mut Frame)>(
//         &self,
//         renderer: &R,
//         bounds: Size,
//         draw_fn: F,
//     ) -> Geometry {
//         renderer.draw_cache(&self.cache, bounds, draw_fn)
//     }

//     fn build_chart<DB: DrawingBackend>(&self, _state: &Self::State, mut chart: ChartBuilder<DB>) {
//         use plotters::prelude::*;

//         const PLOT_LINE_COLOR: RGBColor = RGBColor(0, 175, 255);

//         // Acquire time range
//         let newest_time = self
//             .data_points
//             .front()
//             .unwrap_or(&(DateTime::from_timestamp(0, 0).unwrap(), 0.0))
//             .0;
//         let oldest_time = newest_time - chrono::Duration::seconds(PLOT_SECONDS as i64);
//         let mut chart = chart
//             .x_label_area_size(28)
//             .y_label_area_size(28)
//             .margin(20)
//             .build_cartesian_2d(oldest_time..newest_time, -1f64..1f64)
//             .expect("failed to build chart");

//         chart
//             .configure_mesh()
//             .bold_line_style(plotters::style::colors::BLUE.mix(0.1))
//             .light_line_style(plotters::style::colors::BLUE.mix(0.05))
//             .axis_style(ShapeStyle::from(plotters::style::colors::BLUE.mix(0.45)).stroke_width(1))
//             .x_labels(5)
//             .x_label_style(
//                 ("sans-serif", 12)
//                     .into_font()
//                     .color(&plotters::style::colors::BLUE.mix(0.65))
//                     .transform(FontTransform::Rotate90),
//             )
//             .x_label_formatter(&|x: &DateTime<Utc>| format!("{}", x.timestamp()))
//             .y_labels(10)
//             .y_label_style(
//                 ("sans-serif", 15)
//                     .into_font()
//                     .color(&plotters::style::colors::BLUE.mix(0.65))
//                     .transform(FontTransform::Rotate90),
//             )
//             .y_label_formatter(&|y: &f64| format!("{}%", y))
//             .draw()
//             .expect("failed to draw chart mesh");

//         chart
//             .draw_series(
//                 AreaSeries::new(
//                     self.data_points.iter().map(|x| (x.0, x.1)),
//                     0.0,
//                     PLOT_LINE_COLOR.mix(0.175),
//                 )
//                 .border_style(ShapeStyle::from(PLOT_LINE_COLOR).stroke_width(2)),
//             )
//             .expect("failed to draw chart data");
//     }
// }
