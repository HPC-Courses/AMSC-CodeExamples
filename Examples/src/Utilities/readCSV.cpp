
/*
 * readCSV.cpp
 *
 *  Created on: Jun 25, 2021
 *      Author: forma
 */
#include "readCSV.hpp"
namespace Utility
{
// Implementations.

void
ReadCSV::read(std::basic_istream<CharT> &in)
{
  StringType line;
  unsigned   count = 0u;
  unsigned   maxtokens = 0u;
  unsigned   mintokens = 999999u;
  Token      token;
  Record     tokens;
  StringType word;
  bool       incomplete = false;

  // Reserve space to reduce memory reallocations.
  if(minRecords != 0 && allRecords.capacity() < minRecords)
    allRecords.reserve(minRecords);
  if(minTokens > 1u)
    tokens.reserve(minTokens);

  for(auto i = 0u; i < skip; ++i)
    getline(in, line);

  // Iterate until extraction fails.
  while(std::getline(in, line))
    {
      ++count;
      // Skip empty lines.
      if(line.empty())
        {
          std::cerr << "Warning: empty line " << count << std::endl;
          continue;
        }
      // Trim only complete lines; keep unfinished quoted fields untouched.
      if(!incomplete)
        line = Utility::trim(line);
      std::stringstream s(line);
      while(std::getline(s, word, sep))
        {
          if(!incomplete)
            word = Utility::trim(word);

          if(word.empty())
            {
              tokens.emplace_back();
              continue;
            }

          if(incomplete)
            {
              // End of a quoted token.
              if(word.back() == textMark)
                {
                  if(stripApexes)
                    word.pop_back();
                  token += std::string(1u, sep) + word;
                  tokens.push_back(token);
                  token.clear();
                  incomplete = false;
                }
              else
                token += std::string(1u, sep) + word;
            }
          // Start of a quoted token.
          else if(word.front() == textMark)
            {
              if(stripApexes)
                token = word.substr(1);
              else
                token = word;
              if(token.back() == textMark)
                {
                  if(stripApexes)
                    token.pop_back();
                  tokens.push_back(token);
                  token.clear();
                }
              else
                {
                  incomplete = true;
                }
            }
          else
            {
              token = word;
              tokens.push_back(token);
              token.clear();
            }
        }

      int check{0};
      // Check whether the record contains any non-empty token.
      for(auto const &t : tokens)
        check += t.size();
      if(check != 0)
        {
          if(!incomplete)
            {
              if(tokens.size() < minTokens)
                {
                  if(verbose)
                    {
                      std::clog << "Line " << count << "Incomplete:\n";
                      std::clog << " Has only" << tokens.size()
                                << "Tokens. Autocompleted\n";
                    }
                  for(unsigned int i = 0; i < minTokens - tokens.size(); ++i)
                    {
                      tokens.emplace_back();
                    }
                }
              tokens.shrink_to_fit();
              allRecords.push_back(tokens);
              maxtokens = tokens.size() > maxtokens ? tokens.size() : maxtokens;
              mintokens = tokens.size() < mintokens ? tokens.size() : mintokens;
              tokens.clear();
            }
          else
            {
              std::cerr << "Warning: broken line " << count << std::endl;
            }
        }
      else
        {
          std::cerr << "Warning: empty line " << count << std::endl;
        }
    }
  if(verbose)
    {
      if(in.bad()) std::clog << "Input stream is in a bad state.\n";
      std::clog << " Read " << count << " lines from the file" << std::endl;
      std::clog << " Max/Min number of tokens " << maxtokens << "/" << mintokens
                << std::endl;
      std::clog << " Read " << allRecords.size() << " valid records"
                << std::endl;
      if(!allRecords.empty())
        {
          std::clog << " Last meaningful line read:\n";
          this->writeRecord(std::clog, allRecords.back());
          std::clog << "\n";
        }
    }
  // Reduce memory usage.
  allRecords.shrink_to_fit();
}

void
ReadCSV::writeRecord(std::basic_ostream<CharT> &out, const Record &tokens) const
{
  StringType s;
  StringType separator(1, sep);
  for(auto const &token : tokens)
    s += token + separator;
  s.pop_back();
  out << s;
}

void
ReadCSV::writeAllRecords(std::basic_ostream<CharT> &out) const
{
  for(auto const &t : allRecords)
    {
      writeRecord(out, t);
      out << "\n";
    }
}

} // namespace Utility
