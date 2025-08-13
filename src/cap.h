#include <windows.h>

class capture
{
public:
    capture(int a, int b, int c, int d, LPCSTR j);  // ���ǹ��캯��, ��Ҫָ����ȡ��ͼƬ��С


    LPVOID cap();


private:


    int x;
    int y;
    int w;
    int h;
    BYTE* p;					// ���ݴ洢ָ�룬����Ƶ��������ͷſռ�
    HDC sourceDC;
    HDC momDC;
    HBITMAP memBitmap;
    BITMAPINFOHEADER bih;

    BITMAP BMp;

};

capture::capture(int a, int b, int c, int d, LPCSTR j)
{

    x = (a - c) / 2;
    y = (b - d) / 2;
    w = c;
    h = d;
    LPCSTR jubing = j;
    p = new BYTE[w * h * 3];
    sourceDC = GetDC(FindWindowA(j, NULL));
    momDC = CreateCompatibleDC(sourceDC);
    //memBitmap = CreateCompatibleBitmap(sourceDC, w, h);
    //SelectObject(momDC, memBitmap);


    bih.biSize = sizeof(BITMAPINFOHEADER);
    bih.biBitCount = 24;
    bih.biCompression = 0;
    bih.biHeight = -h;
    bih.biPlanes = 1;
    bih.biWidth = w;

    memBitmap = CreateDIBSection(sourceDC, (LPBITMAPINFO)&bih, DIB_RGB_COLORS, (VOID**)&p, NULL, 0);

    SelectObject(momDC, memBitmap);


}

LPVOID capture::cap()

{

    BitBlt(momDC, 0, 0, w, h, sourceDC, x, y, 0x00CC0020);

    GetObject(memBitmap, sizeof BMp, &BMp);
   

    return BMp.bmBits;

}
