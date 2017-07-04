#!/usr/bin/perl

my $file = shift;
my $dictfile = shift;

my %words = ();
my $line = "";
open(DICT, $dictfile);

while($line = <DICT>)
{
  my @seg = split(/\s+/, $line);
  
  $words{$seg[0]} = 1;
}

close(DICT);

open(SENT, $file);

while($line = <SENT>)
{
  $line =~ s/^\s*//;
  $line =~ s/\s*$//;
  
  next if($line =~ /^\s*$/);
  
  my @seg = split(/\s+/, $line);
  my $str = "";
  my $unk_count = 0;
  foreach my $s (@seg)
  {
    if(defined $words{$s})
    {
      $str .= $s." ";
    }
    else
    {
      $str .= "<unk> ";
      $unk_count++;
    }
  }
  
  if($unk_count <= 1)
  {
    chop($str);
    print $str."\n";
  }
}

close(SENT)
