/*
*
*实现了用OpenCV打开图片，并且将MAT转换成AVFrame
*
*/

//------------- used by named pipe
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>
#include <stdlib.h>
//------------=

#include <iostream>
#include <stdio.h>
#include <string.h>
#include <chrono>
#include <ctime>
#include <thread>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//画质分辨率设置
const int IMAGE_ROW_NUMB = 1080;
const int IMAGE_COL_NUMB = 1920;

/*const int IMAGE_ROW_NUMB = 270;
const int IMAGE_COL_NUMB = 480;*/

#define __STDC_CONSTANT_MACROS

#ifdef _WIN32
//Windows
extern "C" //包含C文件头
{
#include "libavformat/avformat.h"
#include "libavutil/mathematics.h"
#include "libavutil/time.h"
};
#else
//Linux...
#ifdef __cplusplus
extern "C"
{
#endif
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
//#include <libavdevice/avdevice.h>
#include <libavutil/mathematics.h>
#include <linux/videodev2.h>
#include <libavutil/time.h>
#include <SDL/SDL.h>
#ifdef __cplusplus
};
#endif
#endif
#include <iostream>

using namespace std;
using namespace cv;

int flush_encoder(AVFormatContext *ofmt_ctx, unsigned int stream_index, int framecnt)
{
    int ret;
    int got_frame;
    AVPacket enc_pkt;
    if (!(ofmt_ctx->streams[stream_index]->codec->codec->capabilities))
        return 0;
    while (1)
    {
        enc_pkt.data = NULL;
        enc_pkt.size = 0;
        av_init_packet(&enc_pkt);
        ret = avcodec_encode_video2(ofmt_ctx->streams[stream_index]->codec, &enc_pkt,
                                    NULL, &got_frame);
        av_frame_free(NULL);
        if (ret < 0)
            break;
        if (!got_frame)
        {
            ret = 0;
            break;
        }
        printf("Flush Encoder: Succeed to encode 1 frame!\tsize:%5d\n", enc_pkt.size);

        //Write PTS
        AVRational time_base = ofmt_ctx->streams[stream_index]->time_base; //{ 1, 1000 };
        //AVRational r_framerate1 = ifmt_ctx->streams[stream_index]->r_frame_rate;// { 50, 2 };
        AVRational r_framerate1;
        r_framerate1.num = 65535;
        r_framerate1.den = 2733;
        AVRational time_base_q = {1, AV_TIME_BASE};
        //Duration between 2 frames (us)
        int64_t calc_duration = (double)(AV_TIME_BASE) * (1 / av_q2d(r_framerate1)); //内部时间戳
        //Parameters
        enc_pkt.pts = av_rescale_q(framecnt * calc_duration, time_base_q, time_base);
        enc_pkt.dts = enc_pkt.pts;
        enc_pkt.duration = av_rescale_q(calc_duration, time_base_q, time_base);

        /* copy packet*/
        //转换PTS/DTS（Convert PTS/DTS）
        enc_pkt.pos = -1;
        framecnt++;
        ofmt_ctx->duration = enc_pkt.duration * framecnt;

        /* mux encoded frame */
        ret = av_interleaved_write_frame(ofmt_ctx, &enc_pkt);
        if (ret < 0)
            break;
    }
    cvWaitKey(0); //等待用户按键
    //cvReleaseImage(src);//释放图片资源
    cvDestroyWindow("send image"); //释放窗体资源
    return ret;
}

int exit_thread = 0;

int main(int argc, char *argv[])
{
    AVCodecContext *pCodecCtx;
    AVOutputFormat *ofmt = NULL;
    AVFormatContext *ofmt_ctx;
    AVCodec *pCodec;
    AVPacket *dec_pkt, enc_pkt;
    AVFrame *pframe, *pFrameYUV;
    AVStream *video_st;
    int framecnt = 0;
    int videoindex = 0;
    struct SwsContext *img_convert_ctx;
    int ret;
    //旋转角度
    //double angle = 0.9375;
    //推流地址
    const char *out_path = "rtmp://localhost:1935/hls/IPC2_Video";
    //int dec_got_frame;
    int enc_got_frame;
    //ffmpeg注册复用器，编码器等的函数av_register_all()。该函数在所有基于ffmpeg的应用程序中几乎都是第一个被调用的。只有调用了该函数，才能使用复用器，编码器等。
    av_register_all();
    //注册输入设备和输出设备。在使用libavdevice之前，必须先运行avdevice_register_all()对设备进行注册，否则就会出错。
    //avdevice_register_all();
    //加载socket库以及网络加密协议相关的库，为后续使用网络相关提供支持 ,打开网络流的话，前面要加上函数avformat_network_init()
    avformat_network_init();
    //IplImage* src = cvLoadImage("1080P.bmp",1);//加载一张图片
    //Mat src = imread("480.jpg",1);
    // NOTE : set image size
    Mat src = cv::Mat::zeros(IMAGE_ROW_NUMB, IMAGE_COL_NUMB, CV_8UC3);
    AVFrame Frame; // = cvmat_to_avframe(&src);
    //cvNamedWindow("title",1);//创建一个窗体
    //imshow("title", src);//在上面的窗体中显示图片

    //===================================================

    avformat_alloc_output_context2(&ofmt_ctx, NULL, "flv", out_path);
    ofmt = ofmt_ctx->oformat;
    //output encoder initialize
    //avcodec_find_encoder()用于查找FFmpeg的编码器，avcodec_find_decoder()用于查找FFmpeg的解码器。
    pCodec = avcodec_find_encoder(AV_CODEC_ID_H264);
    if (!pCodec)
    {
        printf("Can not find encoder! (没有找到合适的编码器！)\n");
        return -1;
    }
    pCodecCtx = avcodec_alloc_context3(pCodec);
    pCodecCtx->pix_fmt = AV_PIX_FMT_YUV420P;
    pCodecCtx->width = src.cols;
    pCodecCtx->height = src.rows;
    pCodecCtx->time_base.num = 1;
    //帧率设置
    pCodecCtx->time_base.den = 25;
    //pCodecCtx->bit_rate = 400000;
    pCodecCtx->gop_size = 250;

    /* Some formats want stream headers to be separate. */
    if (ofmt_ctx->oformat->flags & AVFMT_GLOBALHEADER)
        pCodecCtx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    //H264 codec param
    //pCodecCtx->me_range = 16;
    //pCodecCtx->max_qdiff = 4;
    //pCodecCtx->qcompress = 0.6;
    pCodecCtx->qmin = 10;
    pCodecCtx->qmax = 51;
    //Optional Param
    //双向预测帧，加强压缩，但是会加大延迟
    pCodecCtx->max_b_frames = 0;
    // Set H264 preset and tune
    AVDictionary *param = 0;
    //av_dict_set(&param, "preset", "fast", 0);
    av_dict_set(&param, "preset", "medium", 0);
    av_dict_set(&param, "tune", "zerolatency", 0);
    if (avcodec_open2(pCodecCtx, pCodec, &param) < 0)
    {
        printf("Failed to open encoder! (编码器打开失败！)\n");
        return -1;
    }
    //Add a new stream to output,should be called by the user before avformat_write_header() for muxing
    //创建流通道avformat_new_stream之后便在 AVFormatContext 里增加了 AVStream 通道（相关的index已经被设置了）。之后，我们就可以自行设置 AVStream 的一些参数信息。例如 : codec_id , format ,bit_rate ,width , height ... ...
    video_st = avformat_new_stream(ofmt_ctx, pCodec);
    if (video_st == NULL)
    {
        return -1;
    }
    video_st->time_base.num = 65535;
    video_st->time_base.den = 2733;
    video_st->codec = pCodecCtx;
    /*
    //Open output URL,set before avformat_write_header() for muxing
    if (avio_open(&ofmt_ctx->pb,out_path, AVIO_FLAG_WRITE) < 0){
    printf("Failed to open output file! (输出文件打开失败！)\n");
    return -1;
    }*/
    //打开输出URL（Open output URL）
    if (!(ofmt->flags & AVFMT_NOFILE))
    {
        //在解码器初始化时，先avio_open创建文件
        ret = avio_open(&ofmt_ctx->pb, out_path, AVIO_FLAG_WRITE);
        if (ret < 0)
        {
            printf("Could not open output URL '%s'", out_path);
            return -1;
        }
    }
    //Show some Information
    av_dump_format(ofmt_ctx, 0, out_path, 1);

    //Write File Header
    //int avformat_write_header(AVFormatContext *s, AVDictionary **options);
    //s：用于输出的AVFormatContext。
    //options：额外的选项，目前没有深入研究过，一般为NULL。
    //函数正常执行后返回值等于0。
    avformat_write_header(ofmt_ctx, NULL);

    //prepare before decode and encode
    dec_pkt = (AVPacket *)av_malloc(sizeof(AVPacket));
    //enc_pkt = (AVPacket *)av_malloc(sizeof(AVPacket));
    //完成输入和输出的初始化之后，就可以正式开始解码和编码并推流的流程了，这里要注意，摄像头数据往往是RGB格式的，需要将其转换为YUV420P格式，所以要先做如下的准备工作
    //camera data has a pix fmt of RGB,convert it to YUV420
    //sws_getContext()：初始化一个SwsContext。成功执行的话返回生成的SwsContext，否则返回NULL。
    img_convert_ctx = sws_getContext(src.cols, src.rows, AV_PIX_FMT_BGR24, pCodecCtx->width, pCodecCtx->height, AV_PIX_FMT_YUV420P, SWS_BICUBIC, NULL, NULL, NULL);
    //用av_frame_alloc(void)函数来分配一个AVFrame结构体。这个函数只是分配AVFrame结构体，但data指向的内存并没有分配，需要我们指定。这个内存的大小就是一张特定格式图像所需的大小，如前一篇博文中说到的，对于YUYV422格式，所需的大小是width * height * 2。
    pFrameYUV = av_frame_alloc();
    uint8_t *out_buffer = (uint8_t *)av_malloc(avpicture_get_size(AV_PIX_FMT_YUV420P, pCodecCtx->width, pCodecCtx->height));
    //使用avpicture_fill来把帧和我们新申请的内存来结合。
    avpicture_fill((AVPicture *)pFrameYUV, out_buffer, AV_PIX_FMT_YUV420P, pCodecCtx->width, pCodecCtx->height);

    printf("\n --------call started----------\n\n");
    //下面就可以正式开始解码、编码和推流了
    //start decode and encode
    //获取启动时间
    int64_t start_time = av_gettime();
    //int av_read_frame(AVFormatContext *s, AVPacket *pkt).
    //ffmpeg中的av_read_frame()的作用是读取码流中的音频若干帧或者视频一帧。例如，解码视频的时候，每解码一个视频帧，需要先调用 av_read_frame()获得一帧视频的压缩数据，然后才能对该数据进行解码（例如H.264中一帧压缩数据通常对应一个NAL）。
    //s：输入的AVFormatContext
    //pkt：输出的AVPacket
    //如果返回0则说明读取正常。
    pFrameYUV->height = pCodecCtx->height;
    pFrameYUV->width = pCodecCtx->width;
    pFrameYUV->format = 0;
    int Fcount = 0;
    int FrameC = 1;
    //char fileName[50];
    char showTex[255];
    time_t t;
    struct tm *lt;

    // init name pipe
    // 通过管道将图片读出
    char *face_named_FIFO = "/tmp/IPC2_Image_Pipe";
    //mkfifo(face_named_FIFO, 0666);
    int face_named_FIFO_fd = open(face_named_FIFO, O_RDONLY);
    //printf("%d",src.size().area() * 3);

    //Frame = cvmat_to_avframe(&src);
    while (1)
    {
        //std::this_thread::sleep_for(1s);
        if (FrameC == 8)
        {
            FrameC = 1;
        }
	    //printf("%s","---++++------");

        //        //--- get image data
        //        src = imread("a1.png",1);
        //        sprintf(fileName,"../../data/a%d.png",FrameC);
        //        printf("\n%s\n",fileName);
        //        src = cv::imread(fileName,1);
        // NOTE : get image from named pipe
        if (read(face_named_FIFO_fd, src.data, src.size().area() * 3) < 1)
        {
	    //printf("%d",src.size().area() * 3);
            printf("data read error!\n");
            continue;
        }
        else
        {
            //cv::namedWindow("send image",0);
            //cv::resizeWindow("send image", IMAGE_COL_NUMB, IMAGE_ROW_NUMB);
            // cv::imshow("send image", src);
            // cv::waitKey(1);
        }

        /*cv::Size src_sz = src.size();
        cv::Size dst_sz(src_sz.height, src_sz.width);
        int len = std::max(src.cols, src.rows);
        //指定旋转中心
        cv::Point2f center(len / 2., len / 2.);

        //获取旋转矩阵（2x3矩阵）
        cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle, 1.0);
        //根据旋转矩阵进行仿射变换
        cv::warpAffine(src, src, rot_mat, dst_sz);*/
        //angle += 0.9375;
        time(&t);           //获取Unix时间戳。
        lt = localtime(&t); //转为时间结构。
        printf("开始编码%d帧%d:%d\n", Fcount, lt->tm_min, lt->tm_sec);
        if (exit_thread)
            break;
        av_log(NULL, AV_LOG_DEBUG, "Going to reencode the frame\n");
        //AVFrame的初始化函数是av_frame_alloc()，销毁函数是av_frame_free()。
        pframe = av_frame_alloc();
        if (!pframe)
        {
            ret = AVERROR(ENOMEM);
            return -1;
        }
        // 水印文字位置
        //cv::Point point(10, 50);
        // 颜色，使用黄色
        //cv::Scalar scalar(0, 255, 255, 0);
        //左上角的信息
        sprintf(showTex, "Frame:%d/%d:%d", Fcount, lt->tm_min, lt->tm_sec);
        cv::putText(src, showTex, Point(20, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 251, 240), 4, 8);
        //av_packet_rescale_ts(dec_pkt, ifmt_ctx->streams[dec_pkt->stream_index]->time_base,
        //	ifmt_ctx->streams[dec_pkt->stream_index]->codec->time_base);
        //ffmpeg中的avcodec_decode_video2()的作用是解码一帧视频数据。输入一个压缩编码的结构体AVPacket，输出一个解码后的结构体AVFrame。
        //（1）对输入的字段进行了一系列的检查工作：例如宽高是否正确，输入是否为视频等等。
        //（2）通过ret = avctx->codec->decode(avctx, picture, got_picture_ptr,&tmp)这句代码，调用了相应AVCodec的decode()函数，完成了解码操作。
        //（3）对得到的AVFrame的一些字段进行了赋值，例如宽高、像素格式等等。
        //ret = avcodec_decode_video2(video_st->codec, pframe,
        //&dec_got_frame, dec_pkt);
        //pframe = &cvmat_to_avframe(&src);

        //SwsContext *sws_getContext(int srcW, int srcH, enum PixelFormat srcFormat, int dstW, int dstH, enum PixelFormat dstFormat, int flags, SwsFilter *srcFilter, SwsFilter *dstFilter, const double *param)
        //總共有十個參數，其中，較重要的是前七個；
        //前三個參數分別代表 source 的寬、高及PixelFormat；
        //四到六個參數分別代表 destination 的寬、高及PixelFormat；
        //第七個參數則代表要使用哪種scale的方法；此參數可用的方法可在 libswscale/swscale.h 內找到。
        //最後三個參數，如無使用，可以都填上NULL。
        //sws_getContext會回傳一個 SwsContext struct，我們可以把這個 struct 看成是個 handler，之後的sws_scale和sws_freeContext皆會用到。
        //pframe->data = Frame.data;
        //memset(pframe->data,0,sizeof(pframe->data));
        //strcpy(pframe->data[0],Frame.data[0]);
        //pframe->data = Frame.data

        //Frame = cvmat_to_avframe(&src);
        //Frame = cvmat_to_avframe(&src);
        avpicture_fill((AVPicture *)&Frame, src.data, AV_PIX_FMT_BGR24, pCodecCtx->width, pCodecCtx->height);
        Frame.width = src.cols;
        Frame.height = src.rows;

        sws_scale(img_convert_ctx, (const uint8_t *const *)Frame.data, Frame.linesize, 0, pCodecCtx->height, pFrameYUV->data, pFrameYUV->linesize);
        enc_pkt.data = NULL;
        enc_pkt.size = 0;
        av_init_packet(&enc_pkt);
        ret = avcodec_encode_video2(pCodecCtx, &enc_pkt, pFrameYUV, &enc_got_frame);
        //ret = avcodec_send_frame(pCodecCtx, pFrameYUV);
        //av_frame_free(&pframe);
        if (enc_got_frame == 1)
        {
            //printf("Succeed to encode frame: %5d\tsize:%5d\n", framecnt, enc_pkt.size);
            framecnt++;
            enc_pkt.stream_index = video_st->index;

            //Write PTS
            AVRational time_base = ofmt_ctx->streams[videoindex]->time_base; //{ 1, 1000 };
            AVRational r_framerate1;                                         //= ifmt_ctx->streams[videoindex]->r_frame_rate;// { 50, 2 };
            r_framerate1.num = 65535;
            r_framerate1.den = 2733;
            AVRational time_base_q = {1, AV_TIME_BASE};
            //Duration between 2 frames (us)
            int64_t calc_duration = (double)(AV_TIME_BASE) * (1 / av_q2d(r_framerate1)); //内部时间戳
            //Parameters
            //而enc_pkt因为是要写入最后的输出码流的，它的PTS、DTS应该是以ofmt_ctx->streams[videoindex]->time_base为时间基来表示的，时间基之间的转换用下式
            //enc_pkt.pts = (double)(framecnt*calc_duration)*(double)(av_q2d(time_base_q)) / (double)(av_q2d(time_base));
            enc_pkt.pts = av_rescale_q(framecnt * calc_duration, time_base_q, time_base);
            enc_pkt.dts = enc_pkt.pts;
            enc_pkt.duration = av_rescale_q(calc_duration, time_base_q, time_base); //(double)(calc_duration)*(double)(av_q2d(time_base_q)) / (double)(av_q2d(time_base));
            enc_pkt.pos = -1;
            //还有一点，因为转码流程可能比实际的播放快很多，为保持流畅的播放，要判断DTS和当前真实时间，并进行相应的延时操作，如下
            //这里正好与之前相反，要将ofmt_ctx->streams[videoindex]->time_base时间基转换为ffmpeg内部时间基，因为av_gettime获得的就是以微秒为单位的时间
            //Delay
            int64_t pts_time = av_rescale_q(enc_pkt.dts, time_base, time_base_q);
            int64_t now_time = av_gettime() - start_time;
            if (pts_time > now_time)
                av_usleep(pts_time - now_time);

            ret = av_interleaved_write_frame(ofmt_ctx, &enc_pkt);
            av_free_packet(&enc_pkt);
            //src.release();
            av_frame_free(&pframe);
            ++Fcount;
            ++FrameC;
        }
        av_free_packet(&enc_pkt);
        //src.release();
        av_frame_free(&pframe);
    }
    //总体流程完毕之后，还剩下最后的flush encoder操作，输出之前存储在缓冲区内的数据
    // 可以看到基本上就是把编码流程重复了一遍
    //至此，就实现了摄像头数据的直播。
    //当然还可以使用多线程来实现“按下回车键停止播放”这样的控制功能。
    //Flush Encoder
    ret = flush_encoder(ofmt_ctx, 0, framecnt);
    if (ret < 0)
    {
        printf("Flushing encoder failed\n");
        return -1;
    }

    //Write file trailer
    av_write_trailer(ofmt_ctx);

    //Clean
    if (video_st)
        avcodec_close(video_st->codec);
    av_free(out_buffer);
    avio_close(ofmt_ctx->pb);
    //avformat_free_context(ifmt_ctx);
    avformat_free_context(ofmt_ctx);
    return 0;
}
