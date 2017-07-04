#!/usr/bin/perl

my $frfile = shift;
my $enfile = shift;
my $batch = shift;
$batch = 64 if length($batch) == 0;

open(FR, $frfile);
open(EN, $enfile);

while(1)
{
  my $line = "";
	for(my $i = 0; $i < $batch; $i++)
	{
		$line = <FR>;
		if($line)
		{	
			print $line;
		}
		else
		{
			last;
		}
	}	

	for(my $i = 0; $i < $batch; $i++)
	{
		$line = <EN>;
		if($line)
		{	
			print $line;
		}
		else
		{
			last;
		}
	}	

   last if ! $line;
}

close(FR);
close(EN);


