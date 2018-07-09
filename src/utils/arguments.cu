#include "arguments.hh"
#include <algorithm>

Arguments::Arguments(const std::vector<std::string>& args)
  : args_(args)
{}

Arguments::Arguments(int argc, char** argv)
{
  for (int i = 0; i < argc; ++i)
    args_.push_back(argv[i]);
}

const std::vector<std::string>& Arguments::args_get() const
{
  return args_;
}

bool Arguments::has_option(char s) const
{
  std::string query = "--";
  query[1] = s;
  return std::find(args_.begin(), args_.end(), query) != args_.end();
}

bool Arguments::has_option(const std::string& l) const
{
  std::string query = "--" + l;
  return std::find(args_.begin(), args_.end(), query) != args_.end();
}

bool Arguments::has_option(char s, const std::string& l) const
{
  return has_option(s) || has_option(l);
}

std::string Arguments::get_option(char s) const
{
  std::string query = "--";
  query[1] = s;
  auto it = std::find(args_.begin(), args_.end(), query);
  if (it == args_.end() || (it + 1) == args_.end())
    return "";
  return *(it + 1);
}
  
std::string Arguments::get_option(const std::string& l) const
{
  std::string query = "--" + l;
  auto it = std::find(args_.begin(), args_.end(), query);
  if (it == args_.end() || (it + 1) == args_.end())
    return "";
  return *(it + 1);
}

std::string Arguments::get_option(char s, const std::string& l) const
{
  auto res = get_option(l);
  return !res.empty() ? res : get_option(s);
}

std::size_t Arguments::size() const
{
  return args_.size();
}

const std::string& Arguments::operator[](std::size_t i) const
{
  return args_[i];
}
