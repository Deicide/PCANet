#include<opencv2/opencv.hpp>
#include<iostream>
#include<fstream>
#include<math.h>
#include<ctime>
#include"utils.h"
#include<omp.h>
using namespace std;
using namespace cv;

void out_file(char *var_name,cv::Mat &var)
{
    char filename[100];
    sprintf(filename,"%s.xml",var_name);
    FileStorage fs(filename,FileStorage::WRITE);
    fs<<var_name<<var;
    fs.release();
}

void print_timestamp(char *name)
{
    time_t t;
    struct tm tmval;
    time(&t);
    tmval = *localtime(&t);
    sprintf(name,"%s%d_%d_%d_%d_%d",name,tmval.tm_mon+1,tmval.tm_mday,tmval.tm_hour,tmval.tm_min,tmval.tm_sec);
}

void out_mat(char *name,cv::Mat &in_mat)
{
    ofstream of(name);
    for(int row = 0 ; row<in_mat.rows; row++)
    {
        for(int col = 0; col<in_mat.cols;col++)
        {
            of<<in_mat.at<double>(row,col)<<'\t';
        }
        of<<endl;
    }
    of.close();
}

///convert each patch from the image into a row vector of the output matrix
///THE PATCH MIGHT CROSS OVER THE RANGE OF THE MATRIX!!!!
///the depth of in_img should be 64F
cv::Mat im2col_step(cv::Mat &in_img,int patch_size,int step)
{

    int out_rows = ((in_img.rows - patch_size)/step + 1)*((in_img.cols - patch_size)/step + 1);
    int out_cols = patch_size * patch_size;
    int r_diff = in_img.rows-patch_size;
    int c_diff = in_img.cols-patch_size;
    int out_row_num=0;
    double *p_out,*p_in;
    cv::Mat out_patches(out_rows,out_cols,CV_64FC1);

    for(int r_offset=0; r_offset <= r_diff; r_offset+=step)
    {
        for(int c_offset=0; c_offset <= c_diff; c_offset+=step)
        {
            p_out = out_patches.ptr<double>(out_row_num);

            for(int r=0;r<patch_size;r++)
            {
                p_in = in_img.ptr<double>(r+r_offset);
                for(int c=0;c<patch_size;c++)
                {
                    p_out[r*patch_size+c] = p_in[c_offset+c];
                }
            }

            out_row_num++;
        }
    }

    /*FileStorage fs("out_patches",FileStorage::WRITE);
    fs<<"out_patches"<<out_patches;
    fs.release();*/

    return out_patches.t();
}

///used to convert the patches of multi channel images into rows.
///listed in the order of RGB
bool im2col_general(cv::Mat &in_img, vector<cv::Mat> &out_vec, int patch_size, int img_size, int step)
{
    int channels=in_img.channels();

    vector<cv::Mat> layers;
    if(channels>1)
        split(in_img,layers);
    else
        layers.push_back(in_img.t());//to be the same as the demo code, where each patch are extracted in column order

    //in RGB order
    for(int i=channels-1;i>=0;i--)
    {
        cv::Mat patch_mat=im2col_step(layers[i], patch_size, 1);
        out_vec.push_back(patch_mat);
    }

    return true;
}

///show the image from the row vector of CIFAR
void show_img(cv::Mat CIFAR_mat)
{
    //NOTICE: opencv uses BGR when showing images.
    IplImage img;
    /*
    uchar temp_data[IMG_DATA_LENGTH];
    for(int i=0; i<1024; i++)
    {
        temp_data[i*3]=CIFAR_mat.data[i+2048];
        temp_data[i*3+1]=CIFAR_mat.data[i+1024];
        temp_data[i*3+2]=CIFAR_mat.data[i];
    }*/
    /*float temp_data[IMG_DATA_LENGTH];
    for(int i=0; i<1024; i++)
    {
        temp_data[i*3]=CIFAR_mat.data[i+2048];
        temp_data[i*3+1]=CIFAR_mat.data[i+1024];
        temp_data[i*3+2]=CIFAR_mat.data[i];
    }
    cv::Mat tmp(IMG_SIZE,IMG_SIZE,CV_32FC3,temp_data);
    img = tmp;*/
    img=CIFAR_mat;
    cvNamedWindow("Image", 1);
    cvShowImage("Image",&img);
    cvWaitKey(0);

    cvDestroyWindow("Image");
    //cvReleaseImage(&&img);
}

void show_multi_img(vector<cv::Mat> img_vec)
{
    //int img_num =
}

int read_CIFAR10(vector<cv::Mat> &train_batch_img, vector<uchar> &train_batch_type, const int stride, bool train)
{
    int sample_toal_num=0;
    int file_num=train? 5:1;

    for(int batch_num=1; batch_num<=file_num; batch_num++)
    {
        char path[100];
        if(train)
            sprintf(path,"/projects/data/cifar-10-batches-bin/data_batch_%d.bin",batch_num);
        else
            //read the test samples
            sprintf(path,"/projects/data/cifar-10-batches-bin/test_batch.bin");

        ifstream batch_file(path,ios::binary);

        //the temporary data
        uchar type;
        uchar tmp[IMG_DATA_LENGTH];
        double *tmpf;
        cv::Mat *im_double;
        //used to subsample
        int sample_num=0;
        while(batch_file.read((char *) &(type),sizeof(char)))
        {

            batch_file.read((char *) tmp,IMG_DATA_LENGTH*sizeof(char));

            if(sample_num++ % stride ==0)
            {
                //convert the scale and data type, and rearrange the order of the pixels
                tmpf= new double[IMG_DATA_LENGTH];
                for(int i=0;i<IMG_DATA_LENGTH/3;i++)
                {
                    tmpf[i*3]=tmp[i+2048];///255.0;
                    tmpf[i*3+1]=tmp[i+1024];///255.0;
                    tmpf[i*3+2]=tmp[i];///255.0;

                }
                //cv::Mat tmp_mat(IMG_SIZE,IMG_SIZE,CV_32FC3,tmpf);
                im_double=new cv::Mat(IMG_SIZE,IMG_SIZE,CV_64FC3,tmpf);

                train_batch_img.push_back(*im_double);
                train_batch_type.push_back(type);


                sample_toal_num++;
            }
        }
        batch_file.close();
    }
        /*FileStorage fs("test_convert1.xml",FileStorage::WRITE);
        fs<<"mat"<<train_batch_img[0];
        fs.release();*/
    //cout<<(int)train_batch_type[70]<<endl;
    //show_img(train_batch_img[70]);


    return sample_toal_num;
}

///convert the pixels from a vector into an image layout
void vec2img(vector<cv::Mat> &CIFAR_data)
{
    uchar temp_data[IMG_DATA_LENGTH];
    int ite_num=IMG_DATA_LENGTH/3;
    int sample_num=CIFAR_data.size();
    for(int k=0;k<sample_num;k++)
    {
        for(int i=0;i<ite_num;i++)
        {
            temp_data[i*3]=CIFAR_data[k].data[i+2048];
            temp_data[i*3+1]=CIFAR_data[k].data[i+1024];
            temp_data[i*3+2]=CIFAR_data[k].data[i];
        }
        cv::Mat tmp(IMG_SIZE,IMG_SIZE,CV_8UC3,temp_data);
        tmp.copyTo(CIFAR_data[k]);
    }
}

///get the PCA filters of each stage
cv::Mat PCA_FilterBank(vector<cv::Mat>& in_img, int patch_size, int num_filters)
{
    //get all the mean-removed patches from the image
    int num_image=in_img.size();
    int size=in_img[0].channels()*patch_size*patch_size;//length of the patch vector
    int in_img_num = in_img.size();
    cv::Mat Rx = cv::Mat::zeros(size,size,in_img[0].depth());
    cv::Mat patch_mean,patch_mean_removed;
    cv::Mat diff_mat;
    vector<cv::Mat> patch_mat_vec;

    int core_num = omp_get_num_procs();
#pragma omp parallel for default(none) num_threads(core_num) private(patch_mat_vec, patch_mean_removed, patch_mean, diff_mat) shared(in_img,patch_size)
    for(int img_num=0; img_num<in_img_num; img_num++)
    {
        //get the patch vector matrix
        im2col_general(in_img[img_num],patch_mat_vec,patch_size,IMG_SIZE,1);
        //remove the mean of patch vectors
        patch_mean_removed.create(0,size,Rx.type());

        for(vector<cv::Mat>::iterator it2=patch_mat_vec.begin();it2!=patch_mat_vec.end();it2++)
        {
            cv::reduce(*it2,patch_mean,0,CV_REDUCE_AVG);

            for(int i=0;i<it2->rows;i++)
            {
                //must use a intermediate mat
                diff_mat=it2->row(i)-patch_mean.row(0);
                patch_mean_removed.push_back(diff_mat);
            }
        }

        //multiply and add the mean-removed patches
        Rx = Rx + patch_mean_removed * patch_mean_removed.t();

        /*FileStorage fs("test_Rx.xml",FileStorage::WRITE);
        fs<<"patch_mean"<<patch_mean<<"patch_mean_removed"<<patch_mean_removed<<"Rx"<<Rx;
        fs.release();*/

        patch_mat_vec.clear();
    }

    Rx=Rx / (double)(num_image*patch_mean.cols);
    //out_mat("stage0_Rx",Rx);
    //out_file("test_Rx",Rx);//delete

    //get the eigen value mat and eigen vector mat of Rx
    cv::Mat e_value_mat,e_vec_mat;
    eigen(Rx,e_value_mat,e_vec_mat);
    //out_mat("e_value_mat",e_value_mat);
    //out_mat("e_vec_mat",e_vec_mat);

    ///you may need to compare the difference of PCA in OpenCV and Matlab
    /*FileStorage fs("test_eigenvector.xml",FileStorage::WRITE);
    fs<<"e_value_mat"<<e_value_mat<<"e_vec_mat_row0"<<e_vec_mat.row(0)<<"e_vec_mat"<<e_vec_mat;
    fs.release();*/

    //the eigen vectors have probably been sorted by eigen value already
    cv::Mat filters(0,size,Rx.depth());
    for(int i=0;i<num_filters;i++)
    {
        filters.push_back(e_vec_mat.row(i));
    }

    return filters;
}

///turn the src_img into the result of convolution
///src_label will be cleared and changed!!!
///for stage 0, the out put is the same as demo; for stage 1, the output is the transposition of demo
vector<cv::Mat> PCA_convolution_rmean(vector<cv::Mat> &src_img, cv::Mat &filter_mat, int patch_size)
{
    const cv::Mat ones_mat=cv::Mat::ones(patch_size*patch_size,1,CV_64FC1);
    cv::Mat padded_img;//zero-padded image which has one channel
    cv::Mat combined_img;//combined all the channels

    int pad=(patch_size-1)/2;
    cv::Scalar s = cv::Scalar(0);

    int channels = src_img[0].channels();
    int out_cols = src_img[0].rows*src_img[0].cols;

    cv::Mat product_mat;
    vector<cv::Mat> single_channel_images;//used to store the split images
    vector<cv::Mat> out_mat;
    vector<uchar> out_label;

    vector<cv::Mat>::iterator src_img_ite=src_img.begin();
    int img_label_num=0;
    int in_img_rows=src_img[0].rows;//suppose all the rows are the same
    while(src_img_ite!=src_img.end())
    {
        //split the 3 channels
        if(channels>1)
            split(*src_img_ite,single_channel_images);
        else
            single_channel_images.push_back(*src_img_ite);

        //turn each channel of image into columns and combine the 3 channels
        cv::Mat temp,patch_mean;
        cv::Mat combiled_img_cols=cv::Mat(0,out_cols,src_img_ite->depth());
        src_img_ite=src_img.erase(src_img_ite);//erase the first one and point to the next position. to save memory.

        //delete
        /*if(stage == 1)
            out_file("after_erase",single_channel_images[0]);*/

        for(vector<cv::Mat>::iterator it=single_channel_images.begin();it!=single_channel_images.end();it++)
        {
            cv::copyMakeBorder(*it,padded_img,pad,pad,pad,pad,cv::BORDER_CONSTANT,s);//zero-padding
            //if(stage==1)
                //padded_img=padded_img.t();//at stage 0, don't need to transform

            temp=im2col_step(padded_img,patch_size,1);//to be the same as the demo
            cv::reduce(temp,patch_mean,0,CV_REDUCE_AVG);
            patch_mean=ones_mat*patch_mean;
            //the mean should be removed
            temp=temp-patch_mean;

            /*FileStorage fs("test_temp.xml",FileStorage::WRITE);
            fs<<"temp"<<temp;
            fs.release();*/
            combiled_img_cols.push_back(temp);
        }
        //delete
            /*if(stage == 1)
                out_file("combiled_img_cols",combiled_img_cols);*/

        for(int filter_num=0;filter_num<filter_mat.rows;filter_num++)
        {
            product_mat =  filter_mat.row(filter_num)* combiled_img_cols;
            product_mat = product_mat.reshape(0,in_img_rows);
            out_mat.push_back(product_mat);
            /*if(stage==0)
                out_mat.push_back(product_mat.t());//to be the same as demo for stage 1
            else
                out_mat.push_back(product_mat);*///no influence. by default: 0.556
            /*if(stage==1)
            {
                out_file("product_mat",product_mat);
                out_file("filter_mat",filter_mat);
            }*/
            /*FileStorage fs("test_convolution.xml",FileStorage::WRITE);
            fs<<"out_mat"<<out_mat[0];
            fs.release();*/
        }
        img_label_num++;
        single_channel_images.clear();
    }

    return out_mat;
}

///modify the input image
bool Heaviside(cv::Mat &in_img)
{
    int row_num=in_img.rows;
    int col_num=in_img.cols;

    double *pt_in_img;

    for(int row=0;row<row_num;row++)
    {
        pt_in_img = in_img.ptr<double>(row);
        for(int col=0;col<col_num;col++)
        {
            pt_in_img[col] = (pt_in_img[col]<0);//to get the same result as the demo
        }
    }

    /*FileStorage fs("test_Heaviside.xml",FileStorage::WRITE);
    fs<<"in_img"<<in_img;
    fs.release();*/
    return true;
}


///each row of block_mat is the vectorized block
cv::Mat Hist(cv::Mat &block_mat, int range)
{
    int row_num=block_mat.rows;//the range of histogram is from 0~range
    cv::Mat BHist = cv::Mat::zeros(row_num,range+1,block_mat.depth());

    double *pt_in,*pt_out;

    for(int row=0;row<row_num;row++)
    {
        pt_in = block_mat.ptr<double>(row);
        pt_out = BHist.ptr<double>(row);
        for(int col=0;col<block_mat.cols;col++)
        {
            pt_out[(int)pt_in[col]]++;
        }
    }

    /*FileStorage fs("test_BHist.xml",FileStorage::WRITE);
    fs<<"BHist"<<BHist;
    fs.release();*/

    BHist=BHist.t();

    return BHist;
}

///again assume that the input images are square
cv::Mat Spp(cv::Mat &BHist, const PCANet PCANet_params, const int img_width)
{
    int start_idx = PCANet_params.histBlockSize[0]/2;
    int end_idx = img_width-PCANet_params.histBlockSize[0]/2;
    int stride = (1-PCANet_params.blkOverlapRatio)*PCANet_params.histBlockSize[0];
    int layers = PCANet_params.pyramid.size();
    //get the dimension of features
    int feature_num = 0;
    for(int i=0;i<layers;i++)
    {
        feature_num += PCANet_params.pyramid[i]*PCANet_params.pyramid[i];
    }
    cv::Mat out_mat = cv::Mat::zeros(BHist.rows,feature_num,BHist.depth());

    //map the features to the corresponding indexes in out_mat
    int idx;
    for(int layer=0;layer<layers;layer++)
    {
        int cnt=0;
        int step = img_width/PCANet_params.pyramid[layer];
        for(int row=start_idx; row<=end_idx; row+=stride)
        {
            for(int col=start_idx; col<=end_idx; col+=stride)
            {

                idx = (row/step)*PCANet_params.pyramid[layer]+col/step;
                out_mat.col(idx) = cv::max(BHist.col(cnt),out_mat.col(idx));
                cnt++;
            }
        }
    }

    /*FileStorage fs("outmat.xml",FileStorage::WRITE);
    fs<<"out_mat"<<out_mat;
    fs.release();*/
    return out_mat;
}

///return value is a row vector of all the histograms
cv::Mat HashingHist(vector<cv::Mat> &in_img, const PCANet PCANet_params)
{
    int num_in_img=in_img.size();
    int num_filters=PCANet_params.numFilters[PCANet_params.numStages-1];
    int num_img0 = num_in_img/num_filters;
    int row_in_img = in_img[0].rows;
    int col_in_img = in_img[0].cols;
    int depth=in_img[0].depth();

    //get the weight constant
    double *w = new double[num_filters];
    w[num_filters-1]=1;
    for(int i=num_filters-2;i>=0;i--)
    {
        w[i]=w[i+1]*2;
    }
    int range = pow(2,num_filters)-1;

    //get the stride constant, assume that the blocks are squares
    int stride = (1-PCANet_params.blkOverlapRatio)*PCANet_params.histBlockSize[0];

    vector<cv::Mat>::iterator it_in_img = in_img.begin();
    cv::Mat BHist;//used to store the histogram
    for(int i=0;i<num_img0;i++)
    {
        cv::Mat T = cv::Mat::zeros(row_in_img,col_in_img,depth);//use higher accuracy to prevent overflow

        for(int j=0;j<num_filters;j++)
        {
            Heaviside(*it_in_img);

            T = T + w[j]* (*it_in_img);
            it_in_img = in_img.erase(it_in_img);
        }

        //out_file("T",T);

        cv::Mat blocks_mat = im2col_step(T,PCANet_params.histBlockSize[0],stride);
        blocks_mat=blocks_mat.t();
        blocks_mat = Hist(blocks_mat,range);
        //out_file("blocks_mat",blocks_mat);
        if(PCANet_params.pyramid.size()!=0)
            blocks_mat = Spp(blocks_mat,PCANet_params,row_in_img);

       // out_file("blocks_mat",blocks_mat);


        if(i==0)
            BHist = blocks_mat;
        else
            hconcat(BHist,blocks_mat,BHist);
        /*FileStorage fs("test_blocksmat.xml",FileStorage::WRITE);
        fs<<"BHist"<<BHist;
        fs.release();*/
    }

    //use spatial pyramid pooling when the parameters are set, and vectorize
    BHist = BHist.reshape(0,1);
    //out_file("BHist",BHist);

    return BHist;
}


///get the patterns of PCA filters
///the label will match the last stage only
PCATrainResult *PCANet_train(vector<cv::Mat> &train_batch_img, vector<uchar> &train_batch_type, const PCANet PCANet_params, bool extract_feature)
{

    PCATrainResult *train_result;
    cv::Mat hashing_result;//each row is a result
    int num_inImg=train_batch_img.size();
    vector<cv::Mat> out_image = train_batch_img;
    vector<uchar> out_type = train_batch_type;
    int channels = train_batch_img[0].channels();

    //allocate storage
    train_result=new PCATrainResult;

    //extract the PCA filters
    for(int stage=0;stage<PCANet_params.numStages;stage++)
    {
        //get the filters
        train_result->filters.push_back(PCA_FilterBank(out_image,PATCH_SIZE,PCANet_params.numFilters[stage]));
        //out_mat("filter1",train_result->filters[0]);
        //do convolution for each intermediate layer
        if(stage != PCANet_params.numStages-1)
            out_image=PCA_convolution_rmean(out_image,train_result->filters[stage],PCANet_params.patchSize[stage]);
    }

    train_batch_img.clear();

    /*FileStorage fs("test_filters.xml",FileStorage::WRITE);
    fs<<"stage1"<<train_result->filters[0]<<"stage2"<<train_result->filters[1];
    fs.release();*/

    if(extract_feature)
    {
        const vector<cv::Mat>::iterator out_img_begin=out_image.begin();
        const vector<uchar>::iterator out_type_begin=out_type.begin();
        int last_stage=PCANet_params.numStages-2;//the array number of last stage
        for(int i=0;i<num_inImg;i++)
        {
            vector<cv::Mat> sub_inImg(out_img_begin+i*PCANet_params.numFilters[last_stage],out_img_begin+(i+1)*PCANet_params.numFilters[last_stage]);

            int size = sub_inImg.size();

            sub_inImg = PCA_convolution_rmean(sub_inImg,train_result->filters[last_stage+1],PCANet_params.patchSize[last_stage+1]);

            if(i==0)
                hashing_result=HashingHist(sub_inImg,PCANet_params);
            else
                hashing_result.push_back(HashingHist(sub_inImg,PCANet_params));

            //out_file("hashing_result",hashing_result);
        }

    }
    //out_file("stage1",train_result->filters[0]);
    //out_file("stage2",train_result->filters[1]);

    //save the result
    train_result->feature = hashing_result;
    //set the label
    float *labels = new float[train_batch_type.size()];
    for(int i=0;i<train_batch_type.size();i++)
        labels[i] = train_batch_type[i];
    cv::Mat temp(train_batch_type.size(),1,CV_32FC1,labels);
    train_result->label=temp;
    train_batch_type.clear();

    return train_result;
}

///SVM parameters are set here
void SVMTrain(cv::Mat &features, cv::Mat &labels, CvSVM &SVM)
{
    cout <<"\n ====== Training Linear SVM Classifier ======= \n"<<endl;
    //set the parameters
    CvSVMParams SVM_params;
    SVM_params.svm_type = CvSVM::C_SVC;
    SVM_params.C = 11;
    SVM_params.kernel_type = CvSVM::LINEAR;
    //SVM_params.term_crit =

    out_mat("features",features);
    out_mat("labels",labels);

    features.convertTo(features,CV_32FC1);
    //create an instance of CvSVM
    int64 e1=cv::getTickCount();
    SVM.train(features,labels,Mat(),Mat(),SVM_params);
    int64 e2=cv::getTickCount();
    float train_time = (e2-e1)/cv::getTickFrequency();
    //delete them
    features.deallocate();

    cout<<"svm training complete, time usage: "<<train_time<<endl;

}
///assume there are 2 stages only
double test_accuracy(vector<cv::Mat> &test_batch_img, vector<uchar> &test_batch_type,
                     vector<cv::Mat> &filters, CvSVM &SVM, PCANet &PCANet_params)
{
    int correct_num=0;
    int total_num=test_batch_img.size();

    vector<cv::Mat>::iterator ite_img =test_batch_img.begin();
    vector<uchar>::iterator ite_label = test_batch_type.begin();
    float prediction;
    while(ite_img!=test_batch_img.end())
    {
        vector<cv::Mat> temp(ite_img,ite_img+1);//used to store one image
        //out_file("stage1_in",temp[0]);

        temp=PCA_convolution_rmean(temp,filters[0],PCANet_params.patchSize[0]);

        temp=PCA_convolution_rmean(temp,filters[1],PCANet_params.patchSize[0]);

        //out_file("stage2_out",temp[0]);


        cv::Mat hashing_result = HashingHist(temp,PCANet_params);
        //make prediction
        hashing_result.convertTo(hashing_result,CV_32FC1);
        prediction = SVM.predict(hashing_result);

        if((uchar)prediction == *ite_label)
            correct_num++;

        ite_img = test_batch_img.erase(ite_img);
        ite_label = test_batch_type.erase(ite_label);
    }

    double accuracy = (double)correct_num/total_num;
    return accuracy;
}

int main(int argc, char **argv)
{
    const int SAMPLE_STRIDE=50;//used for subsampling
    vector<cv::Mat> train_batch_img;
    vector<uchar> train_batch_type;
    vector<cv::Mat> test_batch_img;
    vector<uchar> test_batch_type;
    PCATrainResult *PCA_train_result;
    CvSVM SVM;

    int sample_toal_num;//the total number of samples

    ///set the parameters of PCAnet
    PCANet PCANet_params;
    PCANet_params.numStages=2;
    PCANet_params.numFilters.push_back(40);
    PCANet_params.numFilters.push_back(8);
    PCANet_params.patchSize.push_back(5);
    PCANet_params.patchSize.push_back(5);
    PCANet_params.histBlockSize.push_back(8);
    PCANet_params.histBlockSize.push_back(8);
    PCANet_params.blkOverlapRatio=0.5;
    PCANet_params.pyramid.push_back(4);
    PCANet_params.pyramid.push_back(2);
    PCANet_params.pyramid.push_back(1);

    ///read CIFAR-10 training data
    sample_toal_num=read_CIFAR10(train_batch_img,train_batch_type,SAMPLE_STRIDE,true);
    //show_img(train_batch[0].img);
    cout<<"Use "<<sample_toal_num<<" samples for training."<<endl;

    //PCA_FilterBank(train_batch_img,5,1);//test this function

    PCA_train_result = PCANet_train(train_batch_img,train_batch_type,PCANet_params,true);

    SVMTrain(PCA_train_result->feature,PCA_train_result->label,SVM);
    /*char filename[100]="/projects/data/SVM_params/";
    print_timestamp(filename);
    SVM.save(filename);*/

    sample_toal_num=read_CIFAR10(test_batch_img,test_batch_type,10,false);
    //show_img(train_batch[0].img);
    cout<<"Use "<<sample_toal_num<<" samples for testing."<<endl;

    double accuracy = test_accuracy(test_batch_img,test_batch_type,PCA_train_result->filters,SVM,PCANet_params);

    cout<<accuracy<<endl;

    return 0;
}


