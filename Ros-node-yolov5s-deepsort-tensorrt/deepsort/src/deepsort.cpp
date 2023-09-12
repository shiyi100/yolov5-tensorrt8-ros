#include "deepsort.h"

DeepSort::DeepSort(std::string modelPath, int batchSize, int featureDim, int gpuID, ILogger* gLogger) {
    this->gpuID = gpuID;
    this->enginePath = modelPath;
    this->batchSize = batchSize;
    this->featureDim = featureDim;
    this->imgShape = cv::Size(64, 128);
    this->maxBudget = 100;
    this->maxCosineDist = 0.2;
    this->gLogger = gLogger;
    init();
}

void DeepSort::init() {
    objTracker = new tracker(maxCosineDist, maxBudget);
    featureExtractor = new FeatureTensor(batchSize, imgShape, featureDim, gpuID, gLogger);
    int ret = enginePath.find(".onnx");
    if (ret != -1)
        featureExtractor->loadOnnx(enginePath);
    else
        featureExtractor->loadEngine(enginePath);
}

DeepSort::~DeepSort() {
    delete objTracker;
}

void DeepSort::sort(cv::Mat& frame, vector<DetectBox>& dets) {
    // preprocess Mat -> DETECTION
    DETECTIONS detections;
    vector<CLSCONF> clsConf;
    
    for (DetectBox i : dets) {
        DETECTBOX box(i.x1, i.y1, i.x2-i.x1, i.y2-i.y1);
        DETECTION_ROW d;
        d.tlwh = box;
        d.confidence = i.confidence;
        detections.push_back(d);
        clsConf.push_back(CLSCONF((int)i.classID, i.confidence));
    }
    cout<<"rasfasfwafasdqwd!!!!111111111"<<endl;
    result.clear();
    results.clear();
    cout<<"rasfasfw111111111qwd!!!!111111111"<<endl;
    if (detections.size() > 0) {
          cout<<"rasfasfw77777777777777!!!!111111111"<<endl;
        DETECTIONSV2 detectionsv2 = make_pair(clsConf, detections);
          cout<<"rasf44444444444444444111111111"<<endl;
        sort(frame, detectionsv2);
          cout<<"rasfas9999999999999999911111"<<endl;
    }
    cout<<"rasfasfwafasdqwd!!!!11332222222111"<<endl;
    // postprocess DETECTION -> Mat
    dets.clear();
    for (auto r : result) {
        DETECTBOX i = r.second;
        DetectBox b(i(0), i(1), i(2)+i(0), i(3)+i(1), 1.);
        b.trackID = (float)r.first;
        dets.push_back(b);
    }
    cout<<"rasfasfwa12412412412412432222222111"<<endl;
    for (int i = 0; i < results.size(); ++i) {
        CLSCONF c = results[i].first;
        dets[i].classID = c.cls;
        dets[i].confidence = c.conf;
    }
    cout<<"rasfa13412412412412421124332222222111"<<endl;
}


void DeepSort::sort(cv::Mat& frame, DETECTIONS& detections) {
    bool flag = featureExtractor->getRectsFeature(frame, detections);
    if (flag) {
        objTracker->predict();
        objTracker->update(detections);
        //result.clear();
        for (Track& track : objTracker->tracks) {
            if (!track.is_confirmed() || track.time_since_update > 1)
                continue;
            result.push_back(make_pair(track.track_id, track.to_tlwh()));
        }
    }
}

void DeepSort::sort(cv::Mat& frame, DETECTIONSV2& detectionsv2) {
    std::vector<CLSCONF>& clsConf = detectionsv2.first;
    DETECTIONS& detections = detectionsv2.second;
    cout<<"r44444444444445555555555555555555555554444411111"<<endl;
    bool flag = featureExtractor->getRectsFeature(frame, detections);
    cout<<"r44444777777777777777777777777777777444444411111"<<endl;
    if (flag) {
        objTracker->predict();
        cout<<"r44444444444444444444444444444444444444411111"<<endl;
        objTracker->update(detectionsv2);
        result.clear();
        results.clear();
        for (Track& track : objTracker->tracks) {
            if (!track.is_confirmed() || track.time_since_update > 1)
                continue;
            result.push_back(make_pair(track.track_id, track.to_tlwh()));
            results.push_back(make_pair(CLSCONF(track.cls, track.conf) ,track.to_tlwh()));
        }
    }
}

void DeepSort::sort(vector<DetectBox>& dets) {
    DETECTIONS detections;
    for (DetectBox i : dets) {
        DETECTBOX box(i.x1, i.y1, i.x2-i.x1, i.y2-i.y1);
        DETECTION_ROW d;
        d.tlwh = box;
        d.confidence = i.confidence;
        detections.push_back(d);
    }
    if (detections.size() > 0)
        sort(detections);
    dets.clear();
    for (auto r : result) {
        DETECTBOX i = r.second;
        DetectBox b(i(0), i(1), i(2), i(3), 1.);
        b.trackID = r.first;
        dets.push_back(b);
    }
}

void DeepSort::sort(DETECTIONS& detections) {
    bool flag = featureExtractor->getRectsFeature(detections);
    if (flag) {
        objTracker->predict();
        objTracker->update(detections);
        result.clear();
        for (Track& track : objTracker->tracks) {
            if (!track.is_confirmed() || track.time_since_update > 1)
                continue;
            result.push_back(make_pair(track.track_id, track.to_tlwh()));
        }
    }
}
