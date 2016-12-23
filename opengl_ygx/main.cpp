//
//  main.cpp
//  opengl_ygx
//
//  Created by Ying Gaoxuan on 16/4/2.
//  Copyright © 2016年 Ying Gaoxuan. All rights reserved.
//

#include "cv.h"
#include "highgui.h"

using namespace cv;

#include <vector>
#include <iostream>
#include <fstream>

using namespace std;

#include "GL/freeglut.h"

#include "OGL_OCV_common.h"

#include "MarkerRecognizer.h"

GLuint textureID;

OpenCVGLTexture imgTex;

Mat show_image;

MarkerRecognizer m_recognizer;
Point3f corners_3d[] =
{
    Point3f(-0.5f, -0.5f, 0),
    Point3f(-0.5f,  0.5f, 0),
    Point3f( 0.5f,  0.5f, 0),
    Point3f( 0.5f, -0.5f, 0)
};
float camera_matrix[] =
{
    640.0f, 0.0f, 320.0f,
    0.0f, 640.0f, 240.0f,
    0.0f, 0.0f, 1.0f
};
float dist_coeff[] = {0.0f, 0.0f, 0.0f, 0.0f};
vector<Point3f> m_corners_3d = vector<Point3f>(corners_3d, corners_3d + 4);
Mat m_camera_matrix = Mat(3, 3, CV_32FC1, camera_matrix).clone();
Mat m_dist_coeff = Mat(1, 4, CV_32FC1, dist_coeff).clone();
float m_projection_matrix[16];
float m_model_view_matrix[16];

float ambientLight[]={0.3f,0.3f,0.8f,1.0f};  //环境光
float diffuseLight[]={0.25f,0.25f,0.25f,1.0f}; //散射光
float lightPosition[]={0.0f,0.0f,1.0f,0.0f}; //光源位置
//材质变量
float matAmbient[]={1.0f,1.0f,1.0f,1.0f};
float matDiff[]={1.0f,1.0f,1.0f,1.0f};

void intrinsicMatrix2ProjectionMatrix(cv::Mat& camera_matrix, float width, float height, float near_plane, float far_plane, float* projection_matrix)
{
    float f_x = camera_matrix.at<float>(0,0);
    float f_y = camera_matrix.at<float>(1,1);
    
    float c_x = camera_matrix.at<float>(0,2);
    float c_y = camera_matrix.at<float>(1,2);
    
    projection_matrix[0] = 2*f_x/width;
    projection_matrix[1] = 0.0f;
    projection_matrix[2] = 0.0f;
    projection_matrix[3] = 0.0f;
    
    projection_matrix[4] = 0.0f;
    projection_matrix[5] = 2*f_y/height;
    projection_matrix[6] = 0.0f;
    projection_matrix[7] = 0.0f;
    
    projection_matrix[8] = 1.0f - 2*c_x/width;
    projection_matrix[9] = 2*c_y/height - 1.0f;
    projection_matrix[10] = -(far_plane + near_plane)/(far_plane - near_plane);
    projection_matrix[11] = -1.0f;
    
    projection_matrix[12] = 0.0f;
    projection_matrix[13] = 0.0f;
    projection_matrix[14] = -2.0f*far_plane*near_plane/(far_plane - near_plane);
    projection_matrix[15] = 0.0f;
}

void extrinsicMatrix2ModelViewMatrix(cv::Mat& rotation, cv::Mat& translation, float* model_view_matrix)
{
    
    static double d[] =
    {
        1, 0, 0,
        0, -1, 0,
        0, 0, -1
    };
    Mat_<double> rx(3, 3, d);
    
    rotation = rx*rotation;
    translation = rx*translation;
    
    model_view_matrix[0] = rotation.at<double>(0,0);
    model_view_matrix[1] = rotation.at<double>(1,0);
    model_view_matrix[2] = rotation.at<double>(2,0);
    model_view_matrix[3] = 0.0f;
    
    model_view_matrix[4] = rotation.at<double>(0,1);
    model_view_matrix[5] = rotation.at<double>(1,1);
    model_view_matrix[6] = rotation.at<double>(2,1);
    model_view_matrix[7] = 0.0f;
    
    model_view_matrix[8] = rotation.at<double>(0,2);
    model_view_matrix[9] = rotation.at<double>(1,2);
    model_view_matrix[10] = rotation.at<double>(2,2);
    model_view_matrix[11] = 0.0f;
    
    model_view_matrix[12] = translation.at<double>(0, 0);
    model_view_matrix[13] = translation.at<double>(1, 0);
    model_view_matrix[14] = translation.at<double>(2, 0);
    model_view_matrix[15] = 1.0f;
    
}

void DrawCube(float xPos,float yPos,float zPos, float sScal)
{
    glPushMatrix();
    glScalef(sScal, sScal, sScal);
    glTranslatef(xPos,yPos,zPos);
    glBegin(GL_QUADS);    //顶面
    glNormal3f(0.0f,1.0f,0.0f);
    glVertex3f(0.5f,0.5f,0.5f);
    glVertex3f(0.5f,0.5f,-0.5f);
    glVertex3f(-0.5f,0.5f,-0.5f);
    glVertex3f(-0.5f,0.5f,0.5f);
    glEnd();
    glBegin(GL_QUADS);    //底面
    glNormal3f(0.0f,-1.0f,0.0f);
    glVertex3f(0.5f,-0.5f,0.5f);
    glVertex3f(-0.5f,-0.5f,0.5f);
    glVertex3f(-0.5f,-0.5f,-0.5f);
    glVertex3f(0.5f,-0.5f,-0.5f);
    glEnd();
    glBegin(GL_QUADS);    //前面
    glNormal3f(0.0f,0.0f,1.0f);
    glVertex3f(0.5f,0.5f,0.5f);
    glVertex3f(-0.5f,0.5f,0.5f);
    glVertex3f(-0.5f,-0.5f,0.5f);
    glVertex3f(0.5f,-0.5f,0.5f);
    glEnd();
    glBegin(GL_QUADS);    //背面
    glNormal3f(0.0f,0.0f,-1.0f);
    glVertex3f(0.5f,0.5f,-0.5f);
    glVertex3f(0.5f,-0.5f,-0.5f);
    glVertex3f(-0.5f,-0.5f,-0.5f);
    glVertex3f(-0.5f,0.5f,-0.5f);
    glEnd();
    glBegin(GL_QUADS);    //左面
    glNormal3f(-1.0f,0.0f,0.0f);
    glVertex3f(-0.5f,0.5f,0.5f);
    glVertex3f(-0.5f,0.5f,-0.5f);
    glVertex3f(-0.5f,-0.5f,-0.5f);
    glVertex3f(-0.5f,-0.5f,0.5f);
    glEnd();
    glBegin(GL_QUADS);    //右面
    glNormal3f(1.0f,0.0f,0.0f);
    glVertex3f(0.5f,0.5f,0.5f);
    glVertex3f(0.5f,-0.5f,0.5f);
    glVertex3f(0.5f,-0.5f,-0.5f);
    glVertex3f(0.5f,0.5f,-0.5f);
    glEnd();
    glPopMatrix();
}

void display(void)
{
    
    glEnable2D();
    drawOpenCVImageInGL(imgTex);
    glDisable2D();
    
    glClear(GL_DEPTH_BUFFER_BIT);
    
    intrinsicMatrix2ProjectionMatrix(m_camera_matrix, 640, 480, 0.01f, 100.0f, m_projection_matrix);
    glMatrixMode(GL_PROJECTION);
    static float reflect[] =
    {
        1,  0, 0, 0,
        0, 1, 0, 0,
        0,  0, 1, 0,
        0,  0, 0, 1
    };
    glLoadMatrixf(reflect);
    glMultMatrixf(m_projection_matrix);
    
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    
    
    vector<Marker>& markers = m_recognizer.getMarkers();
    
    m_recognizer.drawToImage(show_image, Scalar(0,255,0), 2);
    imgTex = MakeOpenCVGLTexture(show_image);
    
    Mat r, t;
    for (int i = 0; i < markers.size(); ++i)
    {
        markers[i].estimateTransformToCamera(m_corners_3d, m_camera_matrix, m_dist_coeff, r, t);
        extrinsicMatrix2ModelViewMatrix(r, t, m_model_view_matrix);
        glLoadMatrixf(m_model_view_matrix);
        
        DrawCube(0, 0, -0.5f, 1.0f);
    }
    
    
    glutSwapBuffers();
}

void init_opengl(int argc,char** argv) {
    glutInitWindowSize(640,480);
    glutInitWindowPosition(40,40);
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH); // | GLUT_MULTISAMPLE
    glutCreateWindow("Marker AR");
    
    glClearColor(0.0f,0.0f,0.0f,0.0f);   //清理颜色为黑色
    glShadeModel(GL_SMOOTH);     //使用平滑明暗处理
    glEnable(GL_DEPTH_TEST);     //剔除隐藏面
    glEnable(GL_CULL_FACE);      //不计算多边形背面
    glFrontFace(GL_CCW);      //多边形逆时针方向为正面
    glEnable(GL_LIGHTING);      //启用光照
    //为LIGHT0设置析质
    glMaterialfv(GL_FRONT,GL_AMBIENT,matAmbient);
    glMaterialfv(GL_FRONT,GL_DIFFUSE,matDiff);
    //现在开始调协LIGHT0
    glLightfv(GL_LIGHT0,GL_AMBIENT,ambientLight); //设置环境光分量
    glLightfv(GL_LIGHT0,GL_DIFFUSE,diffuseLight); //设置散射光分量
    glLightfv(GL_LIGHT0,GL_POSITION,lightPosition); //设置光源在场景中的位置
    //启用光
    glEnable(GL_LIGHT0);
    
    glutDisplayFunc(display);
    
}

int start_opengl() {
    
    glutMainLoop();
    
    return 1;
}

int main(int argc, char** argv)
{
    
    cv::VideoCapture capture;
    capture.open(0);
    Mat img;
    
    init_opengl(argc, argv);
  
    while(true)
    {
        glutMainLoopEvent();
        capture >> img;
        show_image = img.clone();
        
        Mat imgTemp = img.clone();
        Mat imgTemp_gray;
        cvtColor(imgTemp, imgTemp_gray, CV_RGBA2GRAY);
        m_recognizer.update(imgTemp_gray, 100);
        
        glutIdleFunc(display);
        glutPostRedisplay();
    }
    
    return 0;
}
