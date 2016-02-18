#include "files.h"
#include<math.h>
class comparision{
  public:
    bool operator() (const pair<float,int> lhs,const pair<float,int> rhs)
    {
      return (lhs.first>rhs.first);
    }
};
int main(int argc,char *argv[]){
  Mat img=imread(argv[1],0);
  Mat vocal;
  Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SURF");
  Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");
  SurfFeatureDetector sif(400);
  BOWImgDescriptorExtractor bowDE(extractor,matcher);
  Mat bowDescriptor;
  FileStorage fs("vocabulary_surf.yml", FileStorage::READ);
  fs["vocabulary_surf"] >> vocal;
  fs.release();
  bowDE.setVocabulary(vocal);
  FileStorage fs2("inverted_index_surf.yml", FileStorage::READ);
  Mat inv,doc_vec;
  fs2["inverted_index_surf"] >> inv;
//  cout<<inv.cols<<" "<<inv.rows<<endl;
  fs2.release();
  vector<KeyPoint> keys;
  vector<vector<int> > v;
  Mat key_desc; 
  sif.detect(img,keys);
  bowDE.compute(img,keys,bowDescriptor,&v,&key_desc);
  DIR *dp;
  struct dirent *dirp;
  struct stat filestat;
  string dir = "./Caltech_11classes",filepath;
  dp = opendir(dir.c_str());
  vector<string> filename;
  while ((dirp = readdir( dp )) )
  {   
    if (strcmp(dirp->d_name,".")==0) continue;
    if (strcmp(dirp->d_name,"..")==0) continue;
    filepath = dir + "/" + dirp->d_name;
    //              cout<<"HI"<<endl;
 //   cout<<filepath<<endl;
    if (stat( filepath.c_str(),&filestat)) continue;
    if (S_ISDIR(filestat.st_mode)){
      DIR *temp_dp;
      struct dirent *temp_dirp;
      struct stat temp_filestat;
      string temp_filename;
      temp_dp=opendir(filepath.c_str());
      int count=0;
      while ((temp_dirp=readdir(temp_dp)) and (count<20)){
	if (strcmp(temp_dirp->d_name,".")==0) continue;
	if (strcmp(temp_dirp->d_name,"..")==0) continue;
	temp_filename=filepath+"/"+temp_dirp->d_name;
	//                              cout<<temp_filename<<endl;
	if (stat(temp_filename.c_str(),&temp_filestat)) continue;
	filename.push_back(temp_filename);
	count++;
     }
      closedir(temp_dp);
    }
  }
  Mat query_vec=Mat_<float>(vocal.rows,1,CV_32FC1);
  for (int i=0;i<vocal.rows;i++){
    query_vec.at<float>(i,0)=v[i].size();
  }
  float  dot_prod[inv.cols];
  for(int j = 0;j<inv.cols;j++)
  {
    float norm=0;
    dot_prod[j]=0;
    for(int i=0;i<inv.rows;i++)
    {
      dot_prod[j] += query_vec.at<float>(i,0) * inv.at<float>(i,j);
      norm+=inv.at<float>(i,j)*inv.at<float>(i,j);
    }
    norm=sqrt(norm);
    dot_prod[j]=dot_prod[j]/norm;
  }
  priority_queue<pair<float,int>,vector< pair<float,int> > > pq;
  for (int i=0;i<inv.cols;i++){
/*    if(dot_prod[i] > maxi)
    {
      index = i;
      maxi = dot_prod[i];
    }*/
    pq.push(make_pair(dot_prod[i],i));
  }
//  printf("%d\n",pq.size());
  for (int i=0;i<8;i++){
  //	Mat ans=imread(filename[pq.top().second]);
        cout<<filename[pq.top().second]<<endl;
	pq.pop();
//  	imshow("Final Answer",ans);
  //	waitKey(0);
  }
  return 0;
}
