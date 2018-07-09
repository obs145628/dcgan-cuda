#pragma once

#include <string>
#include <vector>

class Arguments
{

public:

  Arguments(const std::vector<std::string>& args);
  Arguments(int argc, char** argv);

  const std::vector<std::string>& args_get() const;

  bool has_option(char s) const;
  bool has_option(const std::string& l) const;
  bool has_option(char s, const std::string& l) const;

  std::string get_option(char s) const;
  std::string get_option(const std::string& l) const;
  std::string get_option(char s, const std::string& l) const;

  std::size_t size() const;
  const std::string& operator[](std::size_t i) const;

private:
  std::vector<std::string> args_;
  
};
