#ifndef CAP_H
#define CAP_H

#include <windows.h>
#include <memory>

/**
 * @brief 屏幕捕获类
 */
class capture {
public:
    /**
     * @brief 构造函数
     * @param a 屏幕宽度
     * @param b 屏幕高度
     * @param c 捕获区域宽度
     * @param d 捕获区域高度
     * @param j 窗口标题
     */
    capture(int a, int b, int c, int d, LPCSTR j);

    /**
     * @brief 析构函数，释放所有资源
     */
    ~capture();

    /**
     * @brief 捕获屏幕内容
     * @return 指向捕获数据的指针
     */
    LPVOID cap();

private:
    int x;                      // 捕获区域左上角x坐标
    int y;                      // 捕获区域左上角y坐标
    int w;                      // 捕获区域宽度
    int h;                      // 捕获区域高度
    std::unique_ptr<BYTE[]> p;  // 数据存储指针，使用智能指针自动管理内存
    HDC sourceDC;               // 源设备上下文
    HDC momDC;                  // 内存设备上下文
    HBITMAP memBitmap;          // 内存位图
    BITMAPINFOHEADER bih;       // 位图信息头
    BITMAP BMp;                 // 位图结构
    HWND targetWindow;          // 目标窗口句柄
};

/**
 * @brief 构造函数实现
 */
capture::capture(int a, int b, int c, int d, LPCSTR j) 
    : x((a - c) / 2), y((b - d) / 2), w(c), h(d), sourceDC(nullptr), 
      momDC(nullptr), memBitmap(nullptr), targetWindow(nullptr) {
    
    // 分配内存存储图像数据
    p = std::make_unique<BYTE[]>(w * h * 3);
    
    // 查找目标窗口
    targetWindow = FindWindowA(NULL, j);
    if (targetWindow == NULL) {
        targetWindow = FindWindowA(j, NULL);
    }
    
    // 获取窗口设备上下文
    sourceDC = GetDC(targetWindow);
    if (sourceDC == NULL) {
        // 如果获取失败，使用桌面窗口
        sourceDC = GetDC(GetDesktopWindow());
    }
    
    // 创建兼容的内存设备上下文
    momDC = CreateCompatibleDC(sourceDC);
    
    // 设置位图信息
    bih.biSize = sizeof(BITMAPINFOHEADER);
    bih.biBitCount = 24;
    bih.biCompression = BI_RGB;
    bih.biHeight = -h;
    bih.biPlanes = 1;
    bih.biWidth = w;
    
    // 创建DIB段
    memBitmap = CreateDIBSection(sourceDC, reinterpret_cast<LPBITMAPINFO>(&bih), 
                                 DIB_RGB_COLORS, reinterpret_cast<VOID**>(&p.get()), NULL, 0);
    
    // 选择位图到内存设备上下文
    if (memBitmap != NULL) {
        SelectObject(momDC, memBitmap);
    }
}

/**
 * @brief 析构函数实现，释放所有资源
 */
capture::~capture() {
    // 释放位图资源
    if (memBitmap != NULL) {
        DeleteObject(memBitmap);
        memBitmap = NULL;
    }
    
    // 释放内存设备上下文
    if (momDC != NULL) {
        DeleteDC(momDC);
        momDC = NULL;
    }
    
    // 释放源设备上下文
    if (sourceDC != NULL) {
        ReleaseDC(targetWindow, sourceDC);
        sourceDC = NULL;
    }
}

/**
 * @brief 捕获屏幕内容
 */
LPVOID capture::cap() {
    // 执行位块传输
    if (momDC != NULL && sourceDC != NULL) {
        BitBlt(momDC, 0, 0, w, h, sourceDC, x, y, SRCCOPY);
    }
    
    // 获取位图信息
    if (memBitmap != NULL) {
        GetObject(memBitmap, sizeof(BMp), &BMp);
    }
    
    return BMp.bmBits;
}

#endif // CAP_H
