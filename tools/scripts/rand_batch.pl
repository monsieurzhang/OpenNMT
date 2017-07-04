#!/usr/bin/perl

my $file = shift;
my $batch = shift;
$batch = 64 if length($batch) == 0;

my $line = "";
open(F, $file);

my @batch_sent = ();
my $str = "";
my $sent_count = 0;
while($line = <F>)
{
   if($sent_count < $batch)
   {
      $str .= $line;
      $sent_count++;
   }
   else
   {
      push @batch_sent, $str;
      $str = $line;
      $sent_count = 1;
   }
}

close(F);

my @num = ();

for(my $i = 0; $i < scalar @batch_sent; $i++)
{
  push @num, $i;
}

permutation(\@num);


for(my $i = 0; $i < scalar @num; $i++)
{
  my $cur_batch = $batch_sent[$num[$i]];
 
  #print $num[$i]."\n";
  print $cur_batch;
}

sub permutation 
{  
    my ($array) = @_;  

    my $len = scalar @{$array};
    my $total_swap = $len * 2;
    
    for(my $i = 0; $i < $total_swap; $i++)
    {
      my $r1 = int(rand($len));
      my $r2 = int(rand($len));
      
      if($r1 != $r2)
      {
        my $t = $array->[$r1];
        $array->[$r1] = $array->[$r2];
        $array->[$r2] = $t;
      }
    }

}
