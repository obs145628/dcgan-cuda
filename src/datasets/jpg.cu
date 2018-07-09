#include "jpg.hh"
#include <stdexcept>
#include <setjmp.h>
#include <jpeglib.h>

namespace img
{

    std::uint8_t* jpg_load(const std::string& path,
                           std::size_t* pwidth, std::size_t* pheight,
                           std::size_t* pchannels)
    {
        struct jpeg_decompress_struct cinfo;
        jmp_buf setjmp_buffer;
        struct jpeg_error_mgr pub;

        FILE* infile = fopen(path.c_str(), "rb");
        if (!infile)
            throw std::runtime_error {"Can't open file"};

        cinfo.err = jpeg_std_error(&pub);

        if (setjmp(setjmp_buffer))
            throw std::runtime_error {"Can't read file"};
        
        jpeg_create_decompress(&cinfo);
        jpeg_stdio_src(&cinfo, infile);
        jpeg_read_header(&cinfo, TRUE);
        jpeg_start_decompress(&cinfo);

        std::size_t row_stride = cinfo.output_width * cinfo.output_components;
        std::size_t width = cinfo.output_width;
        std::size_t height = cinfo.output_height;
        std::size_t channels = cinfo.output_components;
        std::uint8_t* data = new std::uint8_t[height * row_stride];

        std::size_t offset = 0;
        while (cinfo.output_scanline < cinfo.output_height)
        {
            auto row = reinterpret_cast<JSAMPROW> (data + offset);
            jpeg_read_scanlines(&cinfo, &row, 1);
            offset += row_stride;
        }

        jpeg_finish_decompress(&cinfo);
        jpeg_destroy_decompress(&cinfo);
        fclose(infile);

        if (pwidth)
            *pwidth = width;
        if (pheight)
            *pheight = height;
        if (pchannels)
            *pchannels = channels;
        return data;
    }

    void jpg_save(const std::string& path, std::uint8_t* data,
                  std::size_t width, std::size_t height,
                  int quality)
    {
        FILE* outfile = fopen(path.c_str(), "wb");
        if (!outfile)
            throw std::runtime_error {"Can't open image file"};

        struct jpeg_compress_struct cinfo;
        struct jpeg_error_mgr       jerr;
 
        cinfo.err = jpeg_std_error(&jerr);
        jpeg_create_compress(&cinfo);
        jpeg_stdio_dest(&cinfo, outfile);

        std::size_t channels = 3;
        J_COLOR_SPACE color_type = JCS_RGB;
 
        cinfo.image_width = width;
        cinfo.image_height = height;
        cinfo.input_components = channels;
        cinfo.in_color_space = color_type;

        jpeg_set_defaults(&cinfo);
        jpeg_set_quality (&cinfo, quality, true);
        jpeg_start_compress(&cinfo, true);

        std::size_t row_stride = width * channels;
        std::size_t offset = 0;
        
        while (cinfo.next_scanline < cinfo.image_height)
        {
            auto row = reinterpret_cast<JSAMPROW> (data + offset);
            jpeg_write_scanlines(&cinfo, &row, 1);
            offset += row_stride;
        }

        jpeg_finish_compress(&cinfo);
    }
    
}
