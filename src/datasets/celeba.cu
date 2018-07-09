#include "celeba.hh"
#include <string>
#include "jpg.hh"


namespace celeba
{

    namespace
    {
        const char* DIR_PATH = "../celeba_norm/";
        constexpr std::size_t IMG_SIZE = 64 * 64 * 3;
    }
    

    dbl_t* load(const std::vector<std::size_t>& idxs)
    {
        dbl_t* res = new dbl_t[IMG_SIZE * idxs.size()];

        for (std::size_t i = 0; i < idxs.size(); ++i)
        {
            dbl_t* img = res + i * IMG_SIZE;
            auto idx = std::to_string(idxs[i]);
            while (idx.size() < 6)
                idx = std::string("0") + idx;
            std::string path = std::string(DIR_PATH) + idx + ".jpg";
            std::uint8_t* pixs = img::jpg_load(path, nullptr, nullptr, nullptr);
            for (std::size_t j = 0; j < IMG_SIZE; ++j)
                img[j] = dbl_t(pixs[j]) / 127.5 - 1.;
            delete[] pixs;
        }
        
        return res;
    }

    dbl_t* load(std::size_t idx_beg, std::size_t idx_end)
    {
        std::vector<std::size_t> idxs;
        for (std::size_t i = idx_beg; i < idx_end; ++i)
            idxs.push_back(i);
        return load(idxs);
    }

    void save_samples(const dbl_t* data, std::size_t width, std::size_t height,
                      const std::string& path)
    {
        std::uint8_t* pixs = new std::uint8_t[width * height * 64 * 64 * 3];
        
        for (std::size_t i = 0; i < width; ++i)
            for (std::size_t j = 0; j < height; ++j)
            {
                const dbl_t* img = data + (j * width + i) * (64 * 64 * 3);
                std::size_t out_x = i * 64;
                std::size_t out_y = j * 64;

                for (std::size_t x = 0; x < 64; ++x)
                    for (std::size_t y = 0; y < 64; ++y)
                        for (std::size_t c = 0; c < 3; ++c)
                        {
                            const dbl_t* pin = img + y * 64 * 3 + x * 3 + c;
                            std::uint8_t* pout = pixs + (out_y + y) * (width * 64 * 3) + (out_x + x) * 3 + c;
                            *pout = (*pin + 1.) * 127.5;
                        }
            }

        img::jpg_save(path, pixs, width * 64, height * 64);
        delete[] pixs;
    }
    
}
