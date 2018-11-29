/***************************************************************
 ***************************************************************/
#pragma once

class Classify;

extern "C"
{
    Classify *GtiClassifyCreate(const char *coefName, const char *labelName);
    void GtiClassifyRelease(Classify *gtiClassify);
    int GtiClassifyFC(Classify *gtiClassify, float *inputData, int count);
    char *GetPredicationString(Classify *gtiClassify, int index);
    int GetPredicationSize(Classify *gtiClassify);
}
