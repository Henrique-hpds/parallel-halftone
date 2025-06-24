#ifndef PPM_H
#define PPM_H

#define RGB_COMPONENT_COLOR 255

typedef struct {
    unsigned char red, green, blue;
} PPMPixel;

typedef struct {
    int x, y;
    PPMPixel *data;
} PPMImage;

#ifdef __cplusplus
extern "C" {
#endif

PPMImage *readPPM(const char *filename);
void writePPM(PPMImage *img, const char* filename);

#ifdef __cplusplus
}
#endif

#endif
